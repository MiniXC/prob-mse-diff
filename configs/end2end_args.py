from dataclasses import dataclass

@dataclass
class End2EndArgs:
    dataset: str = "cdminix/libritts-phones-and-mel"
    train_split: str = "train.other.500+train.clean.360+train.clean.100"
    speakers_in_validation: int = 100
    seed: int = 0
    unseen_validation_split: str = "dev.other+dev.clean+test.other+test.clean"
    wandb_mode: str = "offline"
    wandb_project: str = "protts_evaluation"
    wandb_dir: str = "wandb"
    batch_size: int = 8
    g2p_tokenizer: str = "google/byt5-small"
    g2p_model: str = "models/v5/byt5_speaker"
    encoder_model: str = "models/v5/encoder_diffusion_170k"
    decoder_model: str = "models/v5/decoder_diffusion_64k"
    ddpm_num_steps: int = 1000
    ddpm_beta_schedule: str = "linear"
    steps: int = 40
    enc_length: int = 512
    dec_length: int = 2048
    teacher_force_encoder: bool = False
    teacher_force_decoder: bool = False
    perform_on_unseen_speaker: bool = False
    resynthesis: bool = True
    scale: float = None
    prosody_guidance: float = 1.0