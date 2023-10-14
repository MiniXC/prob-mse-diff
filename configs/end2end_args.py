from dataclasses import dataclass

@dataclass
class End2EndArgs:
    dataset: str = "cdminix/librispeech-phones-and-mel"
    train_split: str = "train.other.500+train.clean.360+train.clean.100"
    speakers_in_validation: int = 100
    seed: int = 0
    unseen_validation_split: str = "dev.other+dev.clean+test.other+test.clean"
    wandb_mode: str = "offline"
    wandb_project: str = "protts_evaluation"
    wandb_dir: str = "wandb"
    batch_size: int = 8
    g2p_tokenizer: str = "google/byt5-small"
    g2p_model: str = "models/byt5_baseline_speaker_32k"
    encoder_model: str = "models/baseline_encoder_diffusion_249k"
    decoder_model: str = "models/baseline_decoder_diffusion_112k"
    ddpm_num_steps: int = 1000
    ddpm_beta_schedule: str = "linear"
    steps: int = 20
    enc_length: int = 512
    dec_length: int = 2048
    teacher_force_encoder: bool = False
    teacher_force_decoder: bool = False
    perform_on_unseen_speaker: bool = False
    resynthesis: bool = True