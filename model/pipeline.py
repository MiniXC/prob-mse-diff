from pathlib import Path

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DDPMScheduler
import imageio
import yaml
from transformers.utils.hub import cached_file

from configs.args import ModelArgs, TrainingArgs


class DDPMPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """
    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler, model_args, training_args, device="cpu"):
        super().__init__()
        # instantiate copy of unet
        self.register_modules(unet=unet, scheduler=scheduler)
        self.model_args = model_args
        self.unet = self.unet.to(device)
        self.unet.eval()
        self._device = device
        self.scale = training_args.diffusion_scale

    @torch.no_grad()
    def __call__(
        self,
        steps,
        phone_cond,
        speaker_cond,
        prosody_cond=None,
        prosody_guidance=1.0,
        batch_size=1,
        generator=None,
        mask=None,
    ):
        if isinstance(self.model_args.sample_size, int):
            image_shape = (
                batch_size,
                self.model_args.in_channels,
                self.model_args.sample_size,
                self.model_args.sample_size,
            )
        else:
            image_shape = (
                batch_size,
                self.model_args.in_channels,
                *self.model_args.sample_size,
            )

        image = randn_tensor(image_shape, device=self._device, generator=generator)

        all_images = []

        self.scheduler.set_timesteps(steps, device=self._device)

        image = image.to(self._device)
        phone_cond = phone_cond.to(self._device)
        speaker_cond = speaker_cond.to(self._device)
        if prosody_cond is not None:
            prosody_cond = prosody_cond.to(self._device)
        mask = mask.to(self._device)

        if self.model_args.model_type == "decoder":
            prosody_mask = torch.rand((batch_size, prosody_cond.shape[1], 1), device=self._device) <= prosody_guidance
            prosody_cond = prosody_cond * prosody_mask

        for t in self.progress_bar(self.scheduler.timesteps):

            # image = self.scheduler.scale_model_input(image, t)

            # 1. predict noise model_output
            if self.model_args.model_type == "encoder":
                model_output = self.unet(
                    image,
                    mask,
                    t,
                    phone_cond,
                    speaker_cond,
                )
            elif self.model_args.model_type == "decoder":
                model_output = self.unet(
                    image,
                    mask,
                    t,
                    phone_cond,
                    speaker_cond,
                    prosody_cond,
                )

            # 2. compute previous image: x_t -> x_t-1
            result = self.scheduler.step(model_output, t, image, generator=generator)
            image = result.prev_sample

            gif_image = result.pred_original_sample.clone().detach().cpu()[0]
            # min-max normalize
            gif_image = (gif_image - gif_image.min()) / (
                gif_image.max() - gif_image.min()
            )
            gif_image = (gif_image * 255).type(torch.uint8)
            gif_image = gif_image.squeeze(0).T
            # flip y axis
            all_images.append(gif_image)

        imageio.mimsave("figures/diffusion_process.gif", all_images, duration=0.05)

        image = image.cpu()
        if self.scale is not None:
            # image = torch.clamp(image, -self.scale, self.scale)
            image = ((image / self.scale) + 1) / 2

        return image

    @staticmethod
    def from_pretrained(path_or_hubid, model, device="cpu", scheduler_class=DDPMScheduler):
        path = Path(path_or_hubid)
        if path.exists():
            config_file = path / "model_config.yml"
            training_config_file = path / "training_args.yml"
        else:
            config_file = cached_file(path_or_hubid, "model_config.yml")
            training_config_file = cached_file(path_or_hubid, "training_args.yml")
        args = yaml.load(open(config_file, "r"), Loader=yaml.Loader)
        args = ModelArgs(**args)
        training_args = yaml.load(open(training_config_file, "r"), Loader=yaml.Loader)
        training_args = TrainingArgs(**training_args)
        scheduler = scheduler_class(
            num_train_timesteps=training_args.ddpm_num_steps,
            beta_schedule=training_args.ddpm_beta_schedule,
            timestep_spacing="linspace",
        )
        pipeline = DDPMPipeline(
            model, 
            scheduler,
            args,
            training_args,
            device=device,
        )
        return pipeline