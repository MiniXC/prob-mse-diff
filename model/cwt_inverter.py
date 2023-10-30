from pathlib import Path
import copy
import math

import yaml
import torch
from torch import nn
from torch.nn import TransformerEncoderLayer
from transformers.utils.hub import cached_file
from rich.console import Console

console = Console()

from configs.cwtargs import ModelArgs

class ConformerLayer(TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        old_kwargs = {k: v for k, v in kwargs.items() if "conv_" not in k}
        super().__init__(*args, **old_kwargs)
        del self.linear1
        del self.linear2
        self.conv1 = nn.Conv1d(
            kwargs["conv_in"],
            kwargs["conv_filter_size"],
            kernel_size=kwargs["conv_kernel"][0],
            padding=(kwargs["conv_kernel"][0] - 1) // 2,
        )
        self.conv2 = nn.Conv1d(
            kwargs["conv_filter_size"],
            kwargs["conv_in"],
            kernel_size=kwargs["conv_kernel"][1],
            padding=(kwargs["conv_kernel"][1] - 1) // 2,
        )

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _ff_block(self, x):
        x = self.conv2(
            self.dropout(self.activation(self.conv1(x.transpose(1, 2))))
        ).transpose(1, 2)
        return self.dropout2(x)

    def _sa_block(
        self,
        x,
        attn_mask,
        key_padding_mask=None,
    ):
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )[0]
        return self.dropout1(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :].to(x.dtype)
        return self.dropout(x)


# from https://pytorch.org/docs/1.13/_modules/torch/nn/modules/transformer.html#TransformerEncoder
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        norm=None,
    ):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask=None,
        src_key_padding_mask=None,
        condition=None,
        return_layer=None,
    ):
        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(
                src_key_padding_mask
            ):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported"
                )

        output = src
        output_for_return = None
        src_key_padding_mask_for_layers = src_key_padding_mask

        if return_layer is not None and return_layer < 0:
            return_layer = self.num_layers + return_layer

        for i, mod in enumerate(self.layers):
            if condition is not None:
                output = output + condition
            output = mod(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask_for_layers,
            )
            if return_layer is not None and i == return_layer:
                output_for_return = output.clone().detach()

        if self.norm is not None:
            output = self.norm(output)

        if return_layer is not None:
            return output, output_for_return
        else:
            return output

class CWTInverterModel(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
    ):
        super().__init__()

        n_widths = args.n_widths
        n_output_bins = 256

        self.duration_in = nn.Sequential(
            nn.Linear(n_widths, args.filter_size),
            nn.ReLU(),
            nn.Linear(args.filter_size, args.filter_size),
            nn.ReLU(),
            nn.Linear(args.filter_size, args.filter_size),
        )

        self.positional_encoding = PositionalEncoding(args.filter_size)

        self.transformer = TransformerEncoder(
            ConformerLayer(
                args.filter_size,
                args.n_heads,
                conv_in=args.filter_size,
                conv_filter_size=args.filter_size,
                conv_kernel=(args.kernel_size, 1),
                batch_first=True,
                dropout=args.dropout,
            ),
            num_layers=args.n_layers,
        )

        self.duration_out = nn.Sequential(
            nn.Linear(args.filter_size, args.filter_size),
            nn.ReLU(),
            nn.Linear(args.filter_size, 1),
        )

        self.apply(self._init_weights)

        self.args = args

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, return_layer=None):
        x = self.duration_in(x)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = self.duration_out(x)
        return x

    def save_model(self, path, accelerator=None):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if accelerator is not None:
            accelerator.save_model(self, path)
        else:
            torch.save(self.state_dict(), path / "pytorch_model.bin")
        with open(path / "model_config.yml", "w") as f:
            f.write(yaml.dump(self.args.__dict__, Dumper=yaml.Dumper))

    @classmethod
    def from_pretrained(cls, path_or_hubid):
        path = Path(path_or_hubid)
        if path.exists():
            config_file = path / "model_config.yml"
            model_file = path / "pytorch_model.bin"
        else:
            config_file = cached_file(path_or_hubid, "model_config.yml")
            model_file = cached_file(path_or_hubid, "pytorch_model.bin")
        args = yaml.load(open(config_file, "r"), Loader=yaml.Loader)
        margs = ModelArgs(**args)
        model = cls(margs)
        model.load_state_dict(torch.load(model_file))
        return model

    @property
    def dummy_input(self):
        torch.manual_seed(0)
        random_input = torch.randn(1, self.args.n_widths, self.args.max_length).transpose(1, 2)
        return random_input