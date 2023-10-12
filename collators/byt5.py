from transformers import AutoTokenizer
import numpy as np
import json

from configs.byt5args import CollatorArgs


class ByT5Collator:
    def __init__(self, args: CollatorArgs):
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        self.max_length = args.max_length
        self.id2phone = json.load(open(args.id2phone_path))

    def __call__(self, b):
        phone_lst = []
        text_lst = []
        speaker_lst = []
        for batch in b:
            phones = np.load(batch["phones"])
            phones = " ".join([self.id2phone[str(p)] for p in phones])
            text = batch["text"]
            speaker = np.load(batch["mean_speaker"])[np.newaxis, :]
            speaker_lst.append(speaker)
            phone_lst.append(phones)
            text_lst.append(text)
        input_result = self.tokenizer(
            text_lst,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.max_length,
            return_tensors="pt",
        )
        label_result = self.tokenizer(
            phone_lst,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.max_length,
            return_tensors="pt",
        )
        speaker_result = np.concatenate(speaker_lst, axis=0)
        return {
            "input_ids": input_result["input_ids"],
            "attention_mask": input_result["attention_mask"],
            "labels": label_result["input_ids"],
            "speaker": speaker_result,
        }
