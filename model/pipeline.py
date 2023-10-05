import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor
import imageio


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

    def __init__(self, unet, scheduler, model_args, timesteps=100, device="cpu"):
        super().__init__()
        # instantiate copy of unet
        self.register_modules(unet=unet, scheduler=scheduler)
        self.model_args = model_args
        self.unet = self.unet.to(device)
        self.unet.eval()
        self._device = device
        self.timesteps = timesteps

    @torch.no_grad()
    def __call__(
        self,
        phone_cond,
        speaker_cond,
        prosody_cond=None,
        speaker_cond_temporal=None,
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

        self.scheduler.set_timesteps(self.timesteps, device=self._device)

        image = image.to(self._device)
        phone_cond = phone_cond.to(self._device)
        speaker_cond = speaker_cond.to(self._device)
        if prosody_cond is not None:
            prosody_cond = prosody_cond.to(self._device)
        if speaker_cond_temporal is not None:
            speaker_cond_temporal = speaker_cond_temporal.to(self._device)
        mask = mask.to(self._device)

        for t in self.progress_bar(self.scheduler.timesteps):
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
                    speaker_cond_temporal,
                )

            print(image[mask].min(), image[mask].max())

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(
                model_output, t, image, generator=generator
            ).prev_sample

            gif_image = image.clone().detach().cpu()[0]
            # min-max normalize
            gif_image = (gif_image - gif_image.min()) / (
                gif_image.max() - gif_image.min()
            )
            gif_image = (gif_image * 255).type(torch.uint8)
            gif_image = gif_image.squeeze(0).T
            # flip y axis
            gif_image = gif_image.flip(0)
            all_images.append(gif_image)

        imageio.mimsave("figures/diffusion_process.gif", all_images, duration=0.05)

        image = image.cpu()

        return image
