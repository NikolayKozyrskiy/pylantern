from typing import TYPE_CHECKING, List

import torch
import torch.nn.functional as F
from matches.loop import Loop

from pylantern.tasks.gan.pix2pix.criterions.general.pix2pixhd_loss import (
    compute_pix2pix_hd_mse,
)
from pylantern.tasks.gan.pix2pix.output_dispatchers.output_dispatcher import (
    BasePix2PixOutputDispatcher,
)

if TYPE_CHECKING:
    from pylantern.tasks.gan.pix2pix.configs import Pix2PixHDConfig
    from pylantern.tasks.gan.pix2pix.pipelines import Pix2PixHDPipeline


class Pix2PixHDOutputDispatcher(BasePix2PixOutputDispatcher):
    def __init__(self, config: "Pix2PixHDConfig", *args, **kwargs):
        BasePix2PixOutputDispatcher.__init__(self, config=config, *args, **kwargs)

    def gan__disc_real_mse(
        self, pipeline: "Pix2PixHDPipeline", loop: "Loop", *args, **kwargs
    ):
        return compute_pix2pix_hd_mse(pipeline.discriminate_dst_image(), target=1.0)

    def gan__disc_fake_mse(
        self, pipeline: "Pix2PixHDPipeline", loop: "Loop", *args, **kwargs
    ):
        return compute_pix2pix_hd_mse(pipeline.discriminate_fake_image(), target=0.0)

    def gan__generator_mse(
        self, pipeline: "Pix2PixHDPipeline", loop: "Loop", *args, **kwargs
    ):
        return compute_pix2pix_hd_mse(
            pipeline.discriminator_src_and_fake_images(), target=1.0
        )

    def gan__gan_feat_loss(
        self, pipeline: "Pix2PixHDPipeline", loop: "Loop", *args, **kwargs
    ):
        loss_G_GAN_Feat = 0.0
        feat_weights = 4.0 / (pipeline.config.n_layers_D + 1)
        D_weights = 1.0 / pipeline.config.num_D
        for i in range(pipeline.config.num_D):
            for j in range(len(pipeline.discriminator_src_and_fake_images()[i]) - 1):
                loss_G_GAN_Feat += F.l1_loss(
                    pipeline.discriminator_src_and_fake_images()[i][j],
                    pipeline.discriminate_dst_image()[i][j].detach(),
                    reduction="mean",
                )
        return D_weights * feat_weights * pipeline.config.lambda_feat * loss_G_GAN_Feat
