from dataclasses import dataclass


@dataclass
class TrainingArgs:
    lr: float = 1e-4
    lr_schedule: str = "linear_with_warmup"
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
    n_steps: int = 25_000
    batch_size: int = 8
    valid_batch_size: int = 8
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
    train_type: str = "byt5"
    use_adafactor: bool = True


@dataclass
class CollatorArgs:
    max_length: int = 768
    tokenizer_name: str = "google/byt5-small"
    id2phone_path: str = "configs/id2phone.json"


@dataclass
class ModelArgs:
    add_speaker_embedding: bool = False
