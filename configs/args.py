from dataclasses import dataclass
from typing import Optional, Tuple, Union


@dataclass
class TrainingArgs:
    lr: float = 8e-5
    lr_schedule: str = "cosine"
    lr_warmup_steps: int = 500
    gradient_clip_val: float = 1.0
    checkpoint_path: str = "checkpoints"
    output_path: str = "outputs"
    load_from_checkpoint: str = None
    run_name: str = None
    wandb_mode: str = "offline"
    wandb_project: str = "protts"
    wandb_dir: str = "wandb"
    train_split: str = "train.other.500+train.clean.360+train.clean.100"
    speakers_in_validation: int = 100
    unseen_validation_split: str = "dev.other+dev.clean+test.other+test.clean"
    n_steps: int = 250_000
    batch_size: int = 16
    valid_batch_size: int = 16
    seed: int = 0
    dataset: str = "cdminix/librispeech-phones-and-mel"
    log_every_n_steps: int = 100
    do_full_eval: bool = True
    do_save: bool = False
    save_onx: bool = False
    eval_only: bool = False
    eval_every_n_steps: int = 1000
    save_every_n_steps: int = 10000
    push_to_hub: bool = False
    hub_repo: str = None
    train_type: str = "encoder"
    ddpm_num_steps: int = 1000
    ddpm_beta_schedule: str = "linear"
    ddpm_num_steps_inference: int = 20
    diffusion_scale: float = 0.7
    loss_type: str = "diffusion" # diffusion or mse
    phone_mask_prob: float = 0.05
    prosody_mask_prob: float = 0.5
    bf16: bool = True


@dataclass
class EncoderCollatorArgs:
    enc_max_length: int = 512
    enc_pack_factor: int = 4
    enc_verbose: bool = False


@dataclass
class DecoderCollatorArgs:
    dec_max_length: int = 2048
    dec_pack_factor: int = 4
    dec_verbose: bool = False


@dataclass
class ModelArgs:
    num_phones: int = 393
    sample_size: int = (512, 80)
    model_type: str = "encoder"
    in_channels: int = 1
    out_channels: int = 1
    center_input_sample: bool = False
    flip_sin_to_cos: bool = True
    freq_shift: int = 0
    down_block_types: Tuple[str] = (
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    )
    mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn"
    up_block_types: Tuple[str] = (
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    )
    only_cross_attention: bool = False
    block_out_channels: Tuple[int] = (
        128,
        128,
        128,
        256,
        512,
    )
    layers_per_block: int = 2
    downsample_padding: int = 1
    mid_block_scale_factor: float = 1
    act_fn: str = "silu"
    norm_num_groups: int = 32
    norm_eps: float = 1e-5
    cross_attention_dim: int = 512
    transformer_layers_per_block: int = 1
    num_attention_heads: int = 8
    dual_cross_attention: bool = False
    use_linear_projection: bool = False
    upcast_attention: bool = False
    resnet_time_scale_shift: str = "default"
    resnet_skip_time_act: bool = False
    resnet_out_scale_factor: int = 1.0
    time_embedding_type: str = "positional"
    time_embedding_dim: int = 512
    time_embedding_act_fn: str = None
    timestep_post_act: str = None
    time_cond_proj_dim: int = None
    conv_in_kernel: int = 3
    conv_out_kernel: int = 3
    projection_class_embeddings_input_dim: int = None
    attention_type: str = "default"
    class_embeddings_concat: bool = False
    mid_block_only_cross_attention: bool = None
    cross_attention_norm: str = None
