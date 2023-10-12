import yaml

from .default import EncoderCollator, DecoderCollator
from .byt5 import ByT5Collator
from configs.args import EncoderCollatorArgs, DecoderCollatorArgs


def get_collator(train_args, collator_args):
    return {
        "encoder": EncoderCollator,
        "decoder": DecoderCollator,
        "byt5": ByT5Collator,
    }[train_args.train_type](collator_args)
