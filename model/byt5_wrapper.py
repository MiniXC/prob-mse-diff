import torch
import torch.nn as nn

from pathlib import Path
from transformers.utils.hub import cached_file
from transformers import T5ForConditionalGeneration
import yaml

from configs.byt5args import ModelArgs


class ByT5Wrapper(nn.Module):
    def __init__(self, args, byt5_model):
        super().__init__()
        if args.add_speaker_embedding:
            self.speaker_projection = nn.Linear(256, 1472)
        self.byt5_model = byt5_model
        self.args = args

    def forward(
        self,
        input_ids,
        labels,
        attention_mask=None,
        speaker=None,
    ):
        input_embeds = self.byt5_model.encoder.embed_tokens(input_ids)
        if self.args.add_speaker_embedding:
            speaker = self.speaker_projection(speaker)
            input_embeds = input_embeds + speaker.unsqueeze(1)
        outputs = self.byt5_model(
            attention_mask=attention_mask,
            inputs_embeds=input_embeds,
            labels=labels,
        )
        return outputs

    def generate(self, input_ids, attention_mask, speaker=None):
        input_embeds = self.byt5_model.encoder.embed_tokens(input_ids)
        if self.args.add_speaker_embedding:
            speaker = self.speaker_projection(speaker)
            input_embeds = input_embeds + speaker.unsqueeze(1)
        outputs = self.byt5_model.generate(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=input_embeds,
            max_length=768,
        )
        return outputs

    def save_model(self, path, accelerator=None):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if accelerator is not None:
            accelerator.save_model(self, path)
        else:
            torch.save(self.state_dict(), path / "pytorch_model.bin")
        with open(path / "model_config.yml", "w") as f:
            f.write(yaml.dump(self.args.__dict__, Dumper=yaml.Dumper))

    @staticmethod
    def from_pretrained(path_or_hubid, pretrained_byt5="google/byt5-small"):
        path = Path(path_or_hubid)
        if path.exists():
            config_file = path / "model_config.yml"
            model_file = path / "pytorch_model.bin"
        else:
            config_file = cached_file(path_or_hubid, "model_config.yml")
            model_file = cached_file(path_or_hubid, "pytorch_model.bin")
        args = yaml.load(open(config_file, "r"), Loader=yaml.Loader)
        args = ModelArgs(**args)
        byt5_model = T5ForConditionalGeneration.from_pretrained(pretrained_byt5)
        model = ByT5Wrapper(args, byt5_model)
        model.load_state_dict(torch.load(model_file))
        return model

    @property
    def dummy_input(self):
        if self.args.add_speaker_embedding:
            return {
                "input_ids": torch.zeros((1, 768), dtype=torch.long),
                "attention_mask": torch.zeros((1, 768), dtype=torch.long),
                "speaker": torch.zeros((1, 256), dtype=torch.float),
                "labels": torch.zeros((1, 768), dtype=torch.long),
            }
        else:
            return {
                "input_ids": torch.zeros((1, 768), dtype=torch.long),
                "attention_mask": torch.zeros((1, 768), dtype=torch.long),
                "labels": torch.zeros((1, 768), dtype=torch.long),
            }
