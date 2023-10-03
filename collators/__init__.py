import yaml

from .default import EncoderCollator, DecoderCollator
from configs.args import EncoderCollatorArgs, DecoderCollatorArgs


def get_collator(train_args, collator_args):
    return {"encoder": EncoderCollator, "decoder": DecoderCollator}[
        train_args.train_type
    ](collator_args)
