from .args import TrainingArgs, ModelArgs, EncoderCollatorArgs, DecoderCollatorArgs


def validate_args(*args):
    for arg in args:
        if isinstance(arg, TrainingArgs):
            if arg.dataset not in ["cdminix/librispeech-phones-and-mel"]:
                raise ValueError(f"dataset {arg.dataset} not supported")
            if arg.lr_schedule not in ["linear_with_warmup", "cosine"]:
                raise ValueError(f"lr_schedule {arg.lr_schedule} not supported")
            if arg.wandb_mode not in ["online", "offline"]:
                raise ValueError(f"wandb_mode {arg.wandb_mode} not supported")
            if arg.wandb_mode == "online":
                if arg.wandb_project is None:
                    raise ValueError("wandb_project must be specified")
            if arg.push_to_hub:
                if arg.hub_repo is None:
                    raise ValueError("hub_repo must be specified")
        if isinstance(arg, ModelArgs):
            # Check inputs
            if len(arg.down_block_types) != len(arg.up_block_types):
                raise ValueError(
                    f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {arg.down_block_types}. `up_block_types`: {arg.up_block_types}."
                )

            if len(arg.block_out_channels) != len(arg.down_block_types):
                raise ValueError(
                    f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {arg.block_out_channels}. `down_block_types`: {arg.down_block_types}."
                )

            if not isinstance(arg.only_cross_attention, bool) and len(
                arg.only_cross_attention
            ) != len(arg.down_block_types):
                raise ValueError(
                    f"Must provide the same number of `only_cross_attention` as `down_block_types`. `only_cross_attention`: {arg.only_cross_attention}. `down_block_types`: {arg.down_block_types}."
                )

            if not isinstance(arg.num_attention_heads, int) and len(
                arg.num_attention_heads
            ) != len(arg.down_block_types):
                raise ValueError(
                    f"Must provide the same number of `num_attention_heads` as `down_block_types`. `num_attention_heads`: {arg.num_attention_heads}. `down_block_types`: {arg.down_block_types}."
                )

            if isinstance(arg.cross_attention_dim, list) and len(
                arg.cross_attention_dim
            ) != len(arg.down_block_types):
                raise ValueError(
                    f"Must provide the same number of `cross_attention_dim` as `down_block_types`. `cross_attention_dim`: {arg.cross_attention_dim}. `down_block_types`: {arg.down_block_types}."
                )

            if not isinstance(arg.layers_per_block, int) and len(
                arg.layers_per_block
            ) != len(arg.down_block_types):
                raise ValueError(
                    f"Must provide the same number of `layers_per_block` as `down_block_types`. `layers_per_block`: {arg.layers_per_block}. `down_block_types`: {arg.down_block_types}."
                )

            if arg.time_embedding_type not in ["fourier", "positional"]:
                raise ValueError(
                    f"{arg.time_embedding_type} does not exist. Please make sure to use one of `fourier` or `positional`."
                )
