import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor


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

    def __init__(self, unet, scheduler, model_type="encoder"):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        self.model_type = model_type

    @torch.no_grad()
    def __call__(
        self,
        phone_cond,
        speaker_cond,
        prosody_cond=None,
        speaker_cond_temporal=None,
        batch_size=1,
        num_inference_steps=1000,
        generator=None,
    ):
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                *self.unet.config.sample_size,
            )

        image = randn_tensor(image_shape, device=self.device, generator=generator)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            if self.model_type == "encoder":
                model_output = self.unet(
                    image,
                    t,
                    phone_cond,
                    speaker_cond,
                )
            elif self.model_type == "decoder":
                model_output = self.unet(
                    image,
                    t,
                    phone_cond,
                    speaker_cond,
                    prosody_cond,
                    speaker_cond_temporal,
                )

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(
                model_output, t, image, generator=generator
            ).prev_sample

        image = image.cpu().permute(0, 2, 3, 1).numpy()

        return image
