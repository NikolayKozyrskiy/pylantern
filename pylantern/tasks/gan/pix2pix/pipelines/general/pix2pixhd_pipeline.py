from typing import TYPE_CHECKING, List, Optional, Union

import torch
from matches.shortcuts.dag import graph_node
from torch import Tensor, nn

from pylantern.common.utils import to_device
from pylantern.tasks.gan.pix2pix.pipelines.pipeline import BasePix2PixPipeline

if TYPE_CHECKING:
    from pylantern.model_zoo.pix2pix import MultiscaleDiscriminator
    from pylantern.tasks.gan.pix2pix.configs import Pix2PixHDConfig
    from pylantern.tasks.gan.pix2pix.models import GeneratorModel


class Pix2PixHDPipeline(BasePix2PixPipeline):
    def __init__(
        self,
        config: "Pix2PixHDConfig",
        generator_model: "GeneratorModel",
        discriminator_model: "MultiscaleDiscriminator",
        device: Union[str, torch.device],
    ):
        BasePix2PixPipeline().__init__(
            self,
            config=config,
            generator_model=generator_model,
            discriminator_model=discriminator_model,
            device=device,
        )
        self.config: "Pix2PixHDConfig"
        self.discriminator_model: "MultiscaleDiscriminator"

    @graph_node
    def discriminate_dst_image(self) -> List[Tensor]:
        pred = self.discriminator_model(
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
        pred = self.discriminator_model(
            torch.concat(
                (
                    self.image_src_normalized(),
                    self.get_predicted_image().detach(),
                ),
                dim=1,
            )
        )
        return pred

    @graph_node
    def discriminator_src_and_fake_images(self) -> List[Tensor]:
        pred = self.discriminator_model(
            torch.concat(
                (
                    self.image_src_normalized(),
                    self.get_predicted_image(),
                ),
                dim=1,
            )
        )
        return pred


def pix2pixhd_pipeline_from_config(
    config: "Pix2PixHDConfig",
    device: Union[str, torch.device],
) -> Pix2PixHDPipeline:
    return Pix2PixHDPipeline(
        config=config,
        generator_model=to_device(config.generator_model(), device=device),
        discriminator_model=to_device(config.discriminator_model(), device=device),
        device=device,
    )
