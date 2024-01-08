from typing import TYPE_CHECKING, List

import torch
import torch.nn.functional as F
from basicsr.losses.gan_loss import r1_penalty
from matches.loop import Loop

from pylantern.tasks.gan.pix2pix.output_dispatchers.output_dispatcher import (
    BasePix2PixOutputDispatcher,
)

if TYPE_CHECKING:
    from pylantern.tasks.gan.pix2pix.configs import GFPGANConfig
    from pylantern.tasks.gan.pix2pix.pipelines import GFPGANPipeline


class GFPGANOutputDispatcher(BasePix2PixOutputDispatcher):
    def __init__(self, config: "GFPGANConfig", *args, **kwargs):
        BasePix2PixOutputDispatcher.__init__(self, config=config, *args, **kwargs)

    def loss__pyramid_reconstruction(
        self, pipeline: "GFPGANPipeline", loop: "Loop", *args, **kwargs
    ):
        pyramid_pred = pipeline.predicted_rgbs_normalized_pyramid()
        pyramid_gt = pipeline.dst_img_normalized_pyramid()
        loss = 0.0
        for i in range(0, pipeline.log_size - 2):
            loss += F.l1_loss(pyramid_pred[i], pyramid_gt[i])

        if (
            loop.iterations.current_epoch
            > pipeline.config.remove_loss_pyramid_reconstruction_epoch
        ):
            loss = loss * 1e-8

        return loss

    def gan__generator(self, pipeline: "GFPGANPipeline", loop: "Loop", *args, **kwargs):
        fake_g_pred = pipeline.discriminate_generator_fake_normalized_input()
        loss = self.complex_criterions.wgan_softplus_loss(
            fake_g_pred, target_is_real=True, is_disc=False
        )
        return loss

    def gan__discriminator_gt(
        self, pipeline: "GFPGANPipeline", loop: "Loop", *args, **kwargs
    ):
        real_d_pred = pipeline.discriminate_discriminator_dst_normalized_input()
        loss = self.complex_criterions.wgan_softplus_loss(
            real_d_pred, target_is_real=True, is_disc=True
        )
        return loss

    def gan__discriminator_fake(
        self, pipeline: "GFPGANPipeline", loop: "Loop", *args, **kwargs
    ):
        fake_g_pred = pipeline.discriminate_discriminator_fake_normalized_input()
        loss = self.complex_criterions.wgan_softplus_loss(
            fake_g_pred, target_is_real=False, is_disc=True
        )
        return loss

    def loss__discriminator_r1_penaly(
        self, pipeline: "GFPGANPipeline", loop: "Loop", *args, **kwargs
    ):
        real_pred = pipeline.discriminate_discriminator_dst_normalized_with_grad()
        gt_img = pipeline.image_dst_normalized_with_grad()
        loss = (
            r1_penalty(real_pred, gt_img)
            * pipeline.config.discriminator_r1_penalty_every_iter
            / 2.0
            + 0.0 * real_pred[0]
        )
        return loss

    def loss__vgg19_perceptual_basicsr(
        self, pipeline: "GFPGANPipeline", loop: "Loop", *args, **kwargs
    ):
        loss, loss_style = self.complex_criterions.vgg19_perceptual_basicsr(
            x=pipeline.get_predicted_image(),
            gt=pipeline.image_dst_normalized(),
        )
        return loss + loss_style
