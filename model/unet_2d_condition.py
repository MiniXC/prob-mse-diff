# adapted from https://raw.githubusercontent.com/huggingface/diffusers/v0.20.0/src/diffusers/models/unet_2d_condition.py
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
import yaml
from pathlib import Path
from transformers.utils.hub import cached_file
from diffusers import UNet2DModel
from diffusers.models.activations import get_activation
from diffusers.models.embeddings import (
    GaussianFourierProjection,
    PositionNet,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.unet_2d_blocks import (
    UNetMidBlock2DCrossAttn,
    UNetMidBlock2DSimpleCrossAttn,
    get_down_block,
    get_up_block,
)

from configs.args import ModelArgs


class WrapperUNet2DConditionModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model = UNet2DModel(
            sample_size=args.sample_size,
            in_channels=2,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(
                64,
                64,
                128,
                256,
                512,
            ),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                # "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                # "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        self.phone_embedding = nn.Embedding(args.num_phones, 80)
        self.args = args

    def forward(
        self,
        sample,
        sample_mask,
        timesteps,
        phone_cond,
        speaker_cond,
        prosody_cond=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        phone_emb = self.phone_embedding(phone_cond)
        phone_emb = phone_emb.reshape(sample.shape[0], 1, -1, 80)
        sample = torch.cat([sample, phone_emb], dim=1)
        return self.model(
            sample,
            timesteps,
        ).sample

    def save_model(self, path, accelerator=None):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if accelerator is not None:
            accelerator.save_model(self, path)
        else:
            torch.save(self.state_dict(), path / "pytorch_model.bin")
        with open(path / "model_config.yml", "w") as f:
            f.write(yaml.dump(self.args.__dict__, Dumper=yaml.Dumper))

    @staticmethod
    def from_pretrained(path_or_hubid):
        path = Path(path_or_hubid)
        if path.exists():
            config_file = path / "model_config.yml"
            model_file = path / "pytorch_model.bin"
        else:
            config_file = cached_file(path_or_hubid, "model_config.yml")
            model_file = cached_file(path_or_hubid, "pytorch_model.bin")
        args = yaml.load(open(config_file, "r"), Loader=yaml.Loader)
        args = ModelArgs(**args)
        model = WrapperUNet2DConditionModel(args)
        model.load_state_dict(torch.load(model_file))
        return model

    @property
    def dummy_input(self):
        torch.manual_seed(0)
        if self.args.model_type == "encoder":
            return [
                torch.randn(1, 1, *self.args.sample_size),
                torch.ones(1, 1, self.args.sample_size[0]),
                torch.randint(0, 100, (1,)),
                torch.randint(
                    0,
                    100,
                    (
                        1,
                        self.args.sample_size[0],
                    ),
                ),
                torch.randn(1, self.args.sample_size[0]),
            ]
        elif self.args.model_type == "decoder":
            return [
                torch.randn(1, 1, *self.args.sample_size),
                torch.ones(1, 1, self.args.sample_size[0]),
                torch.randint(0, 100, (1,)),
                torch.randn(1, self.args.sample_size[0]),
            ]


class CustomUNet2DConditionModel(nn.Module):
    r"""
    A conditional 2D UNet model that takes a noisy sample, conditional state, and a timestep and returns a sample
    shaped output.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.sample_size = args.sample_size

        # input
        conv_in_padding = (args.conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2d(
            args.in_channels,
            args.block_out_channels[0] // 2,
            kernel_size=args.conv_in_kernel,
            padding=conv_in_padding,
        )
        self.conv_in_cond = nn.Conv2d(
            args.in_channels,
            args.block_out_channels[0] // 2,
            kernel_size=args.conv_in_kernel,
            padding=conv_in_padding,
        )

        # time
        if args.time_embedding_type == "fourier":
            time_embed_dim = args.time_embedding_dim or args.block_out_channels[0] * 2
            if time_embed_dim % 2 != 0:
                raise ValueError(
                    f"`time_embed_dim` should be divisible by 2, but is {time_embed_dim}."
                )
            self.time_proj = GaussianFourierProjection(
                time_embed_dim // 2,
                set_W_to_weight=False,
                log=False,
                flip_sin_to_cos=args.flip_sin_to_cos,
            )
            timestep_input_dim = time_embed_dim
        elif args.time_embedding_type == "positional":
            time_embed_dim = args.time_embedding_dim or args.block_out_channels[0] * 4

            self.time_proj = Timesteps(
                args.block_out_channels[0], args.flip_sin_to_cos, args.freq_shift
            )
            timestep_input_dim = args.block_out_channels[0]

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=args.act_fn,
            post_act_fn=args.timestep_post_act,
            cond_proj_dim=args.time_cond_proj_dim,
        )

        if args.model_type == "encoder":
            self.phone_embedding = nn.Embedding(args.num_phones, 80)
            self.speaker_embedding = nn.Sequential(
                nn.Linear(256, 80),
                nn.GELU(),
                nn.Linear(80, 80),
            )
        elif args.model_type == "decoder":
            self.phone_embedding = nn.Embedding(args.num_phones, 80)
            self.speaker_embedding = nn.Sequential(
                nn.Linear(256, 80),
                nn.GELU(),
                nn.Linear(80, 80),
            )
            self.speaker_embedding_temporal = nn.Sequential(
                nn.Linear(40, 80),
                nn.GELU(),
                nn.Linear(80, 80),
            )
            self.prosody_embedding = nn.Sequential(
                nn.Linear(80, 80),
                nn.GELU(),
                nn.Linear(80, 80),
            )

        if args.time_embedding_act_fn is None:
            self.time_embed_act = None
        else:
            self.time_embed_act = get_activation(args.time_embedding_act_fn)

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        if isinstance(args.only_cross_attention, bool):
            if args.mid_block_only_cross_attention is None:
                args.mid_block_only_cross_attention = args.only_cross_attention

            args.only_cross_attention = [args.only_cross_attention] * len(
                args.down_block_types
            )

        if args.mid_block_only_cross_attention is None:
            args.mid_block_only_cross_attention = False

        if isinstance(args.num_attention_heads, int):
            args.num_attention_heads = (args.num_attention_heads,) * len(
                args.down_block_types
            )

        if isinstance(args.cross_attention_dim, int):
            args.cross_attention_dim = (args.cross_attention_dim,) * len(
                args.down_block_types
            )

        if isinstance(args.layers_per_block, int):
            args.layers_per_block = [args.layers_per_block] * len(args.down_block_types)

        if isinstance(args.transformer_layers_per_block, int):
            args.transformer_layers_per_block = [
                args.transformer_layers_per_block
            ] * len(args.down_block_types)

        blocks_time_embed_dim = time_embed_dim

        # down
        output_channel = args.block_out_channels[0]
        for i, down_block_type in enumerate(args.down_block_types):
            input_channel = output_channel
            output_channel = args.block_out_channels[i]
            is_final_block = i == len(args.block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=args.layers_per_block[i],
                transformer_layers_per_block=args.transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=blocks_time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=args.norm_eps,
                resnet_act_fn=args.act_fn,
                resnet_groups=args.norm_num_groups,
                cross_attention_dim=args.cross_attention_dim[i],
                num_attention_heads=args.num_attention_heads[i],
                downsample_padding=args.downsample_padding,
                dual_cross_attention=args.dual_cross_attention,
                use_linear_projection=args.use_linear_projection,
                only_cross_attention=args.only_cross_attention[i],
                upcast_attention=args.upcast_attention,
                resnet_time_scale_shift=args.resnet_time_scale_shift,
                attention_type=args.attention_type,
                resnet_skip_time_act=args.resnet_skip_time_act,
                resnet_out_scale_factor=args.resnet_out_scale_factor,
                cross_attention_norm=args.cross_attention_norm,
                attention_head_dim=args.num_attention_heads[i]
                if args.num_attention_heads[i] is not None
                else output_channel,
            )
            self.down_blocks.append(down_block)

        # mid
        if args.mid_block_type == "UNetMidBlock2DCrossAttn":
            self.mid_block = UNetMidBlock2DCrossAttn(
                transformer_layers_per_block=args.transformer_layers_per_block[-1],
                in_channels=args.block_out_channels[-1],
                temb_channels=blocks_time_embed_dim,
                resnet_eps=args.norm_eps,
                resnet_act_fn=args.act_fn,
                output_scale_factor=args.mid_block_scale_factor,
                resnet_time_scale_shift=args.resnet_time_scale_shift,
                cross_attention_dim=args.cross_attention_dim[-1],
                num_attention_heads=args.num_attention_heads[-1],
                resnet_groups=args.norm_num_groups,
                dual_cross_attention=args.dual_cross_attention,
                use_linear_projection=args.use_linear_projection,
                upcast_attention=args.upcast_attention,
                attention_type=args.attention_type,
            )
        elif args.mid_block_type == "UNetMidBlock2DSimpleCrossAttn":
            self.mid_block = UNetMidBlock2DSimpleCrossAttn(
                in_channels=args.block_out_channels[-1],
                temb_channels=blocks_time_embed_dim,
                resnet_eps=args.norm_eps,
                resnet_act_fn=args.act_fn,
                output_scale_factor=args.mid_block_scale_factor,
                cross_attention_dim=args.cross_attention_dim[-1],
                attention_head_dim=args.num_attention_heads[-1],
                resnet_groups=args.norm_num_groups,
                resnet_time_scale_shift=args.resnet_time_scale_shift,
                skip_time_act=args.resnet_skip_time_act,
                only_cross_attention=args.mid_block_only_cross_attention,
                cross_attention_norm=args.cross_attention_norm,
            )
        elif args.mid_block_type is None:
            self.mid_block = None
        else:
            raise ValueError(f"unknown mid_block_type : {args.mid_block_type}")

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(args.block_out_channels))
        reversed_num_attention_heads = list(reversed(args.num_attention_heads))
        reversed_layers_per_block = list(reversed(args.layers_per_block))
        reversed_cross_attention_dim = list(reversed(args.cross_attention_dim))
        reversed_transformer_layers_per_block = list(
            reversed(args.transformer_layers_per_block)
        )
        only_cross_attention = list(reversed(args.only_cross_attention))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(args.up_block_types):
            is_final_block = i == len(args.block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(args.block_out_channels) - 1)
            ]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=reversed_layers_per_block[i] + 1,
                transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=blocks_time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=args.norm_eps,
                resnet_act_fn=args.act_fn,
                resnet_groups=args.norm_num_groups,
                cross_attention_dim=reversed_cross_attention_dim[i],
                num_attention_heads=reversed_num_attention_heads[i],
                dual_cross_attention=args.dual_cross_attention,
                use_linear_projection=args.use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=args.upcast_attention,
                resnet_time_scale_shift=args.resnet_time_scale_shift,
                attention_type=args.attention_type,
                resnet_skip_time_act=args.resnet_skip_time_act,
                resnet_out_scale_factor=args.resnet_out_scale_factor,
                cross_attention_norm=args.cross_attention_norm,
                attention_head_dim=args.num_attention_heads[i]
                if args.num_attention_heads[i] is not None
                else output_channel,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if args.norm_num_groups is not None:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=args.block_out_channels[0],
                num_groups=args.norm_num_groups,
                eps=args.norm_eps,
            )

            self.conv_act = get_activation(args.act_fn)

        else:
            self.conv_norm_out = None
            self.conv_act = None

        conv_out_padding = (args.conv_out_kernel - 1) // 2
        self.conv_out = nn.Conv2d(
            args.block_out_channels[0],
            args.out_channels,
            kernel_size=args.conv_out_kernel,
            padding=conv_out_padding,
        )

        if args.attention_type == "gated":
            positive_len = 768
            if isinstance(args.cross_attention_dim, int):
                positive_len = args.cross_attention_dim
            elif isinstance(args.cross_attention_dim, tuple) or isinstance(
                args.cross_attention_dim, list
            ):
                positive_len = args.cross_attention_dim[0]
            self.position_net = PositionNet(
                positive_len=positive_len, out_dim=args.cross_attention_dim
            )

        self.args = args

    def forward(
        self,
        sample,
        sample_mask,
        timesteps,
        phone_cond,
        speaker_cond,
        prosody_cond=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        padding_mask = sample_mask.unsqueeze(-1)
        upsample_size = None

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        if encoder_attention_mask is not None:
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(sample.dtype)
            ) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        if self.args.center_input_sample:
            sample = 2 * sample - 1.0

        if not torch.is_tensor(timesteps):
            raise ValueError(f"timesteps should be a tensor, but is {type(timesteps)}.")
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)

        if self.args.model_type == "encoder":
            phone_emb = self.phone_embedding(phone_cond)
            speaker_emb = self.speaker_embedding(speaker_cond)
            cond = phone_emb + speaker_emb
        elif self.args.model_type == "decoder":
            phone_emb = self.phone_embedding(phone_cond)
            speaker_emb = self.speaker_embedding(speaker_cond)
            prosody_emb = self.prosody_embedding(prosody_cond)
            cond = phone_emb + speaker_emb + prosody_emb

        cond = cond.unsqueeze(1)

        emb = self.time_embedding(t_emb, None)

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        sample = sample * padding_mask
        cond = cond * padding_mask
        sample = self.conv_in(sample)
        cond = self.conv_in_cond(cond)

        # combine sample and cond by concatenating along the channel dimension
        sample = torch.cat([sample, cond], dim=1)

        down_block_res_samples = (sample,)

        for downsample_block in self.down_blocks:
            if (
                hasattr(downsample_block, "has_cross_attention")
                and downsample_block.has_cross_attention
            ):
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
            )

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            # if not is_final_block and forward_upsample_size:
            #     upsample_size = down_block_res_samples[-1].shape[2:]
            if (
                hasattr(upsample_block, "has_cross_attention")
                and upsample_block.has_cross_attention
            ):
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        sample = sample * padding_mask

        return sample

    def save_model(self, path, accelerator=None):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if accelerator is not None:
            accelerator.save_model(self, path)
        else:
            torch.save(self.state_dict(), path / "pytorch_model.bin")
        with open(path / "model_config.yml", "w") as f:
            f.write(yaml.dump(self.args.__dict__, Dumper=yaml.Dumper))

    @staticmethod
    def from_pretrained(path_or_hubid):
        path = Path(path_or_hubid)
        if path.exists():
            config_file = path / "model_config.yml"
            model_file = path / "pytorch_model.bin"
        else:
            config_file = cached_file(path_or_hubid, "model_config.yml")
            model_file = cached_file(path_or_hubid, "pytorch_model.bin")
        args = yaml.load(open(config_file, "r"), Loader=yaml.Loader)
        args = ModelArgs(**args)
        model = CustomUNet2DConditionModel(args)
        model.load_state_dict(torch.load(model_file))
        return model

    @property
    def dummy_input(self):
        torch.manual_seed(0)
        if self.args.model_type == "encoder":
            return [
                torch.randn(1, 1, *self.sample_size),
                torch.ones(1, 1, self.sample_size[0]),
                torch.randint(0, 100, (1,)),
                torch.randint(
                    0,
                    100,
                    (
                        1,
                        self.sample_size[0],
                    ),
                ),
                torch.randn(1, self.sample_size[0], 256),
            ]
        elif self.args.model_type == "decoder":
            return [
                torch.randn(1, 1, *self.sample_size),
                torch.ones(1, 1, self.sample_size[0]),
                torch.randint(0, 100, (1,)),
                torch.randn(1, self.sample_size[0], 256),
                torch.randn(1, self.sample_size[0], 80),
            ]
