from typing import Sequence, Union

from pylantern.common.constants import DEFAULT_IMG_MEAN, DEFAULT_IMG_STD
from pylantern.model_zoo.gfpgan import (
    StyleGAN2Discriminator,
    gfpgan_generator,
    stylegan2_discriminator,
)
from pylantern.tasks.gan.pix2pix.configs.config import BasePix2PixConfig
from pylantern.tasks.gan.pix2pix.models import (
    GeneratorInferenceModel,
    GeneratorModel,
    get_generator_model,
)


class GFPGANConfig(BasePix2PixConfig):
    num_style_feat: int = 512
    fix_decoder: bool = True
    channel_multiplier: int = 1
    resample_kernel: Sequence[int] = (1, 3, 3, 1)
    num_mlp: int = 8
    lr_mlp: float = 0.01
    sft_half: bool = True
    input_is_latent: bool = True
    different_w: bool = True
    stddev_group: int = 4
    narrow: float = 1.0
    remove_loss_pyramid_reconstruction_epoch: int = 20
    discriminator_r1_penalty_every_iter: int = 16

    def generator_model(
        self, *args, **kwargs
    ) -> Union["GeneratorModel", "GeneratorInferenceModel"]:
        return get_generator_model(
            generator=gfpgan_generator(
                arch="orig",
                predict_mask=self.predict_mask,
                out_size=self.image_size[0],
                decoder_load_path=None,  # Path("_d/gfpgan/weights/StyleGAN2_512_Cmul1_FFHQ_B12G4_scratch_800k.pth"),
                fix_decoder=self.fix_decoder,
                num_style_feat=self.num_style_feat,
                channel_multiplier=self.channel_multiplier,
                resample_kernel=self.resample_kernel,
                num_mlp=self.num_mlp,
                lr_mlp=self.lr_mlp,
                input_is_latent=self.input_is_latent,
                different_w=self.different_w,
                narrow=self.narrow,
                sft_half=self.sft_half,
            ),
            mean=DEFAULT_IMG_MEAN,
            std=DEFAULT_IMG_STD,
            input_range=[-1, 1],
        )

    def discriminator_model(self, *args, **kwargs) -> "StyleGAN2Discriminator":
        return stylegan2_discriminator(
            out_size=self.image_size[0],
            channel_multiplier=self.channel_multiplier,
            resample_kernel=self.resample_kernel,
            stddev_group=self.stddev_group,
            narrow=self.narrow,
        )
