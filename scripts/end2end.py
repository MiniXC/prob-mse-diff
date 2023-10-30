import os
import sys
from pathlib import Path
import json

sys.path.append(".")  # add root of project to path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchaudio
import numpy as np

from transformers import HfArgumentParser, AutoTokenizer, AutoModelForMaskedLM
from datasets import load_dataset

from configs.end2end_args import End2EndArgs
from model.byt5_wrapper import ByT5Wrapper
from model.unet_2d_condition import CustomUNet2DConditionModel
from model.pipeline import DDPMPipeline
from diffusers import DDPMScheduler, EulerAncestralDiscreteScheduler
from PIL import Image
from simple_hifigan import Synthesiser
from collators.default import EncoderCollator, DecoderCollator

from rich.console import Console

console = Console()


def prepare_for_g2p(item, tokenizer):
    tokenizer_result = tokenizer(
        [item["text"]],
        return_tensors="pt",
    )
    speaker = np.load(item["mean_speaker"])
    speaker = torch.from_numpy(speaker).unsqueeze(0)
    dict = {
        "input_ids": tokenizer_result["input_ids"],
        "attention_mask": tokenizer_result["attention_mask"],
        "speaker": speaker,
    }
    return dict


def prepare_for_encoder(item, phones, phone2id, lm_model, lm_tokenizer):
    phones = [int(phone2id[p]) for p in phones[0].split(" ")]
    phones = np.array(phones)
    phones = torch.from_numpy(phones).unsqueeze(0)
    speaker = np.load(item["mean_speaker"])
    speaker = torch.from_numpy(speaker).unsqueeze(0)
    # repeat speaker so that shape is (1, phone_len, speaker_dim)
    speaker = speaker.repeat(phones.shape[1], 1).unsqueeze(0)
    # pad both phones and speaker to args.enc_length
    # first, create mask
    mask = torch.ones((1, args.enc_length), dtype=torch.bool)
    mask[:, phones.shape[1] :] = False
    # then, pad phones and speaker
    phones = F.pad(phones, (0, args.enc_length - phones.shape[1]))
    speaker = F.pad(speaker, (0, 0, 0, args.enc_length - speaker.shape[1]))

    if lm_model is not None:
        text = item["text"]
        lm_inputs = lm_tokenizer(
            [text],
            return_tensors="pt",
            max_length=128,
            pad_to_multiple_of=128,
            truncation=True,
        )
        with torch.no_grad():
            lm_cond = lm_model(**lm_inputs, output_hidden_states=True).hidden_states[-1]
            lm_mask = lm_inputs["attention_mask"]
    else:
        lm_cond = None
        lm_mask = None

    dict = {
        "phones": phones,
        "speaker": speaker,
        "mask": mask,
        "lm_cond": lm_cond,
        "lm_mask": lm_mask,
    }
    return dict


def prepare_for_decoder(item, item_for_encoder, prosody):
    prosody = prosody.numpy()
    duration = prosody[:, 30]
    # denormalize
    duration = np.ceil(2 ** (duration * 11)).astype(np.int32)
    print(duration)

    phones = item_for_encoder["phones"][0].numpy()
    speaker = item_for_encoder["speaker"][0].numpy()
    mask = item_for_encoder["mask"][0].numpy()
    phones = phones[: duration.shape[0]]
    speaker = speaker[: duration.shape[0]]
    mask = mask[: duration.shape[0]]
    prosody = prosody[: duration.shape[0]]
    # repeat all according to duration (including prosody)
    phones = np.repeat(phones, duration, axis=0)
    speaker = np.repeat(speaker, duration, axis=0)
    mask = np.repeat(mask, duration, axis=0)
    prosody = np.repeat(prosody, duration, axis=0)
    # to torch
    phones = torch.from_numpy(phones)
    speaker = torch.from_numpy(speaker)
    mask = torch.from_numpy(mask)
    prosody = torch.from_numpy(prosody)
    # pad all to args.dec_length
    # first, create mask
    mask = torch.ones((1, args.dec_length), dtype=torch.bool)
    mask[:, phones.shape[0] :] = False
    # then, pad phones and speaker
    phones = F.pad(phones, (0, args.dec_length - phones.shape[0]))
    speaker = F.pad(speaker, (0, 0, 0, args.dec_length - speaker.shape[0]))
    mask = F.pad(mask, (0, args.dec_length - mask.shape[1]))
    prosody = F.pad(prosody, (0, 0, 0, args.dec_length - prosody.shape[0]))
    # batch dimension
    phones = phones.unsqueeze(0)
    speaker = speaker.unsqueeze(0)
    prosody = prosody.unsqueeze(0)

    dict = {
        "phones": phones,
        "speaker": speaker,
        "mask": mask,
        "prosody": prosody,
    }
    return dict


def main():
    global args

    parser = HfArgumentParser([End2EndArgs])
    args = parser.parse_args_into_dataclasses()[0]

    # Load the models
    console.rule("Loading models")
    tokenizer = AutoTokenizer.from_pretrained(args.g2p_tokenizer)
    g2p_model = ByT5Wrapper.from_pretrained(args.g2p_model)
    encoder_model = CustomUNet2DConditionModel.from_pretrained(args.encoder_model)
    decoder_model = CustomUNet2DConditionModel.from_pretrained(args.decoder_model)
    if encoder_model.args.lm_condition is not None:
        lm_model = AutoModelForMaskedLM.from_pretrained(encoder_model.args.lm_condition)
        lm_tokenizer = AutoTokenizer.from_pretrained(encoder_model.args.lm_condition)
    else:
        lm_model = None
        lm_tokenizer = None
    synth = Synthesiser()
    g2p_model.eval()
    encoder_model.eval()
    decoder_model.eval()
    id2phone = json.load(open("configs/id2phone.json"))
    phone2id = {v: k for k, v in id2phone.items()}
    console.print("Models loaded")

    # Load the dataset
    if not args.perform_on_unseen_speaker:
        train_ds = load_dataset(args.dataset, split=args.train_split)
        np.random.seed(args.seed)
        speakers = np.random.choice(
            train_ds.unique("speaker_id"),
            size=args.speakers_in_validation,
            replace=False,
        )
        dataset = train_ds.filter(
            lambda x: x["speaker_id"] in speakers and np.random.rand() < 0.5,
            keep_in_memory=True,
        )
    else:
        dataset = load_dataset(args.dataset, split=args.unseen_validation_split)

    first_item = dataset[0]
    first_item[
        "text"
    ] = "Hi Gustav, this is another test of the Text-to-Speech system. I hope you like it."
    first_item_g2p = prepare_for_g2p(first_item, tokenizer)

    console.rule("Running G2P")
    console.print(f"Input text: [green]{first_item['text']}[/green]")
    console.print(
        f"Speaker: [green]{Path(first_item['mean_speaker']).parent.name}[/green]"
    )
    g2p_result = g2p_model.generate(
        first_item_g2p["input_ids"],
        first_item_g2p["attention_mask"],
        first_item_g2p["speaker"],
    )
    g2p_result = tokenizer.batch_decode(g2p_result, skip_special_tokens=True)
    console.print(f"G2P result: {g2p_result}")
    # convert to phones

    g2p_result = ["<s0> h aɪ ɡ ʌ s t æ v ð ɪ s ɪ z ɐ n ʌ ð ɚ t ɛ s t ʌ v ð ə t iː t iː ɛ s s ɪ s t ə m <s1> a ɪ h o ʊ p j uː l aɪ k ɪ t <s0>"]
    first_item_encoder = prepare_for_encoder(first_item, g2p_result, phone2id, lm_model, lm_tokenizer)

    if args.teacher_force_encoder:
        teacher_forced = EncoderCollator.item_to_arrays(first_item)
        tf_phones = torch.from_numpy(teacher_forced[1].astype(np.int32))
        tf_speaker = torch.from_numpy(teacher_forced[2])
        tf_phones_len = tf_phones.shape[0]
        tf_phones = tf_phones.unsqueeze(0)
        tf_mask = torch.ones((1, args.dec_length), dtype=torch.bool)
        tf_mask[:, tf_phones_len:] = False
        first_item_decoder["phones"][0, :] = 0
        first_item_decoder["phones"][0, :tf_phones_len] = tf_phones
        first_item_decoder["speaker"][0, :] = 0
        first_item_decoder["speaker"][0, :tf_phones_len] = tf_speaker
        first_item_decoder["mask"] = tf_mask

    console.rule("Running encoder")
    encoder_pipeline = DDPMPipeline.from_pretrained(
        args.encoder_model,
        encoder_model,
        device="cpu",
    )
    encoder_result = encoder_pipeline(
        args.steps,
        first_item_encoder["phones"],
        first_item_encoder["speaker"],
        batch_size=1,
        mask=first_item_encoder["mask"],
        lm_cond=first_item_encoder["lm_cond"],
        lm_mask=first_item_encoder["lm_mask"],
    )[0][0]
    encoder_result = encoder_result[first_item_encoder["mask"][0]]
    encoder_result = torch.clamp(encoder_result, 0, 1)
    img = Image.fromarray((encoder_result.detach().numpy() * 255).astype(np.uint8).T)
    img.save("figures/encoder_result.png")

    console.rule("Running decoder")
    first_item_decoder = prepare_for_decoder(
        first_item, first_item_encoder, encoder_result
    )

    if args.teacher_force_decoder:
        teacher_forced = DecoderCollator.item_to_arrays(first_item)
        tf_prosody = torch.from_numpy(teacher_forced[0])
        tf_phones = torch.from_numpy(teacher_forced[1].astype(np.int32))
        tf_speaker = torch.from_numpy(teacher_forced[2])
        tf_phones_len = tf_phones.shape[0]
        tf_phones = tf_phones.unsqueeze(0)
        tf_mask = torch.ones((1, args.dec_length), dtype=torch.bool)
        tf_mask[:, tf_phones_len:] = False
        first_item_decoder["prosody"][0, :] = 0
        first_item_decoder["prosody"][0, :tf_phones_len] = tf_prosody
        first_item_decoder["phones"][0, :] = 0
        first_item_decoder["phones"][0, :tf_phones_len] = tf_phones
        first_item_decoder["speaker"][0, :] = 0
        first_item_decoder["speaker"][0, :tf_phones_len] = tf_speaker
        first_item_decoder["mask"] = tf_mask

    decoder_pipeline = DDPMPipeline.from_pretrained(
        args.decoder_model,
        decoder_model,
        device="cpu",
    )
    print(
        first_item_decoder["phones"].min(), first_item_decoder["phones"].max(), "phones"
    )
    print(
        first_item_decoder["speaker"].min(),
        first_item_decoder["speaker"].max(),
        "speaker",
    )
    print(
        first_item_decoder["prosody"].min(),
        first_item_decoder["prosody"].max(),
        "prosody",
    )
    decoder_result = decoder_pipeline(
        args.steps,
        first_item_decoder["phones"],
        first_item_decoder["speaker"],
        first_item_decoder["prosody"],
        batch_size=1,
        mask=first_item_decoder["mask"],
        prosody_guidance=args.prosody_guidance,
    )[0][0]
    decoder_result = decoder_result[first_item_decoder["mask"][0]]
    decoder_result = torch.clamp(decoder_result, 0, 1)
    img = Image.fromarray((decoder_result.detach().numpy() * 255).astype(np.uint8).T)
    img.save("figures/decoder_result.png")

    # denormalize mel
    decoder_result = decoder_result.numpy()
    mel_range = (-11, 2)
    mel_denorm = (decoder_result * (mel_range[1] - mel_range[0])) + mel_range[0]

    # flip
    print(mel_denorm.shape)
    mel_denorm = np.flip(mel_denorm, axis=1)
    print(mel_denorm.shape)

    audio = synth(mel_denorm.copy())
    audio = torch.from_numpy(audio)
    torchaudio.save("figures/audio.wav", audio, 22050)

    if args.resynthesis:
        mel = DecoderCollator.item_to_arrays(first_item)[3]
        mel = (mel * (mel_range[1] - mel_range[0])) + mel_range[0]
        mel = np.flip(mel, axis=1)
        audio = synth(mel.copy())
        audio = torch.from_numpy(audio)
        torchaudio.save("figures/audio_original.wav", audio, 22050)


if __name__ == "__main__":
    main()
