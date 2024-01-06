from typing import TYPE_CHECKING, List, Optional, Union

import torch
from matches.shortcuts.dag import graph_node
from torch import Tensor, nn

from pylantern.common.utils import to_device

from .pipeline import GanPix2PixPipeline

if TYPE_CHECKING:
    from pylantern.model_zoo.gfpgan import FacialComponentDiscriminator
    from pylantern.model_zoo.pix2pix import MultiscaleDiscriminator

    from ..configs import Pix2PixHDConfig


class Pix2PixHDPipeline(GanPix2PixPipeline):
    def __init__(
        self,
        config: "Pix2PixHDConfig",
        generator: "nn.Module",
        discriminator: "MultiscaleDiscriminator",
        device: Union[str, torch.device],
        discriminator_left_eye: Optional["FacialComponentDiscriminator"] = None,
        discriminator_right_eye: Optional["FacialComponentDiscriminator"] = None,
        discriminator_mouth: Optional["FacialComponentDiscriminator"] = None,
    ):
        super().__init__(
            config=config,
            generator=generator,
            discriminator=discriminator,
            device=device,
            discriminator_left_eye=discriminator_left_eye,
            discriminator_right_eye=discriminator_right_eye,
            discriminator_mouth=discriminator_mouth,
        )
        self.config: "Pix2PixHDConfig"
        self.discriminator: "MultiscaleDiscriminator"

    @graph_node
    def discriminate_dst_image(self) -> List[Tensor]:
        pred = self.discriminator(
            torch.concat(
                (
                    self.image_src_normalized(),
                    self.image_dst_normalized().detach(),
                ),
                dim=1,
            )
        )
        return pred

    @graph_node
    def discriminate_fake_image(self) -> List[Tensor]:
        pred = self.discriminator(
            torch.concat(
                (
                    self.image_src_normalized(),
                    self.face_swapper_generate_image().detach(),
                ),
                dim=1,
            )
        )
        return pred

    @graph_node
    def discriminator_src_and_fake_images(self) -> List[Tensor]:
        pred = self.discriminator(
            torch.concat(
                (
                    self.image_src_normalized(),
                    self.face_swapper_generate_image(),
                ),
                dim=1,
            )
        )
        return pred


def pix2pix_pipeline_from_config(
    config: "Pix2PixHDConfig",
    device: Union[str, torch.device],
) -> Pix2PixHDPipeline:
    return Pix2PixHDPipeline(
        config=config,
        generator=to_device(config.avaturn_swapper_model(), device=device),
        discriminator=to_device(config.discriminator_model(), device=device),
        discriminator_left_eye=to_device(
            config.discriminator_left_eye_model(), device=device
        ),
        discriminator_right_eye=to_device(
            config.discriminator_right_eye_model(), device=device
        ),
        discriminator_mouth=to_device(
            config.discriminator_mouth_model(), device=device
        ),
        device=device,
    )
