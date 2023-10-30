from dataclasses import dataclass

@dataclass
class ModelArgs:
    n_widths: int = 9
    filter_size: int = 256
    n_layers: int = 4
    kernel_size: int = 3
    dropout: float = 0.1
    n_heads: int = 4
    max_length: int = None # this is set by the trainer, changing this value will have no effect