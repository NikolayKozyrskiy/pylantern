from typing import TYPE_CHECKING, List

import torch
import torch.nn.functional as F
from matches.loop import Loop

from pylantern.common.nn.functional import gram_matrix, mse_reduction
from pylantern.output_dispatcher import BaseOutputDispatcher
from pylantern.tasks.gan.pix2pix.criterions.general import (
    complex_criterions as base_pix2pix_cc,
)

if TYPE_CHECKING:
    from pylantern.tasks.gan.pix2pix.configs import BasePix2PixConfig
    from pylantern.tasks.gan.pix2pix.pipelines import BasePix2PixPipeline


class BasePix2PixOutputDispatcher(BaseOutputDispatcher):
    def __init__(self, config: "BasePix2PixConfig", *args, **kwargs):
        BaseOutputDispatcher.__init__(
            self,
            config=config,
            complex_criterions_module=base_pix2pix_cc,
            *args,
            **kwargs
        )

    def gan__temporal_mse(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        cropped_orig_predict = pipeline.random_cropped_orig_0875_predict()
        cropped_predict = pipeline.random_cropped_0875_predict()
        coeff = int(loop.iterations.current_epoch) / pipeline.config.max_epoch
        coeff = coeff ** (1.0 / 4.0)
        loss = (
            mse_reduction(cropped_orig_predict, cropped_predict, reduction="mean")
            * coeff
        )
        return loss

    def gan__temporal_l1(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        cropped_orig_predict = pipeline.random_cropped_orig_0875_predict()
        cropped_predict = pipeline.random_cropped_0875_predict()
        coeff = int(loop.iterations.current_epoch) / pipeline.config.max_epoch
        coeff = coeff ** (1.0 / 4)
        loss = (
            F.l1_loss(cropped_orig_predict, cropped_predict, reduction="mean") * coeff
        )
        return loss

    def gan__temporal_l1_smooth(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        cropped_orig_predict = pipeline.random_cropped_orig_0875_predict()
        cropped_predict = pipeline.random_cropped_0875_predict()
        coeff = int(loop.iterations.current_epoch) / pipeline.config.max_epoch
        coeff = coeff ** (1.0 / 4)
        loss = (
            F.smooth_l1_loss(
                cropped_orig_predict, cropped_predict, beta=0.01, reduction="mean"
            )
            * coeff
        )
        return loss

    def gan__temporal_ms_dssim(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        cropped_orig_predict = pipeline.random_cropped_orig_0875_predict()
        cropped_predict = pipeline.random_cropped_0875_predict()
        coeff = int(loop.iterations.current_epoch) / pipeline.config.max_epoch
        coeff = coeff ** (1.0 / 4)
        loss = (
            self.complex_criterions.ms_dssim(cropped_orig_predict, cropped_predict)
            * coeff
        )
        return loss

    def gan__temporal_laplacian_pyramid(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        cropped_orig_predict = pipeline.random_cropped_orig_0875_predict()
        cropped_predict = pipeline.random_cropped_0875_predict()
        coeff = int(loop.iterations.current_epoch) / pipeline.config.max_epoch
        coeff = coeff ** (1.0 / 4)
        loss = (
            self.complex_criterions.laplacian_pyramid(
                cropped_orig_predict, cropped_predict
            )
            * coeff
        )
        return loss

    def gan__vision_aided_discriminator(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        loss = self.complex_criterions.vision_aided_discriminator(
            pipeline.image_dst_normalized(), for_real=True
        ).mean()
        loss = (
            loss
            + self.complex_criterions.vision_aided_discriminator(
                pipeline.get_predicted_image(), for_real=False
            ).mean()
        )
        return loss

    def gan__vision_aided_generator(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        loss = self.complex_criterions.vision_aided_discriminator(
            pipeline.get_predicted_image(), for_G=True
        ).mean()
        return loss

    def loss__vgg_perceptual(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        loss = (
            self.complex_criterions.vgg19_perceptual(
                pipeline.get_predicted_image_destandardized(),
                pipeline.image_dst_scaled().detach(),
            )
            * pipeline.config.lambda_feat
        )
        return loss

    def loss__vgg19_perceptual_and_style(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        loss = self.complex_criterions.vgg19_perceptual_and_style(
            pipeline.get_predicted_image_destandardized(),
            pipeline.image_dst_scaled().detach(),
        )
        return loss

    def loss__lpips_perceptual(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        loss = self.complex_criterions.lpips_alex_perceptual(
            pipeline.get_predicted_image(),
            pipeline.image_dst_normalized(),
        ).mean()
        return loss

    def loss__dreamsim_cos_ensemble_perceptual(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        pred = self.complex_criterions.dreamsim_ensemble.embed(
            pipeline.get_predicted_image_resized_destandardized_224()
        )
        gt = self.complex_criterions.dreamsim_ensemble.embed(
            pipeline.image_dst_resized_scaled_224(),
        ).detach()
        return (1 - F.cosine_similarity(pred, gt, dim=-1)).mean()

    def loss__bce_with_logits_mask(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        loss = F.binary_cross_entropy_with_logits(
            pipeline.segmentation_mask_predict(),
            pipeline.segmentation_mask_dst(),
            reduction="mean",
        )
        return loss

    def loss__sobel(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        loss = self.complex_criterions.sobel(
            pipeline.get_predicted_image_destandardized(),
            pipeline.image_dst_scaled(),
        )
        return loss

    def loss__recon_ms_dssim(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        loss = self.complex_criterions.ms_dssim(
            pipeline.get_predicted_image_destandardized(),
            pipeline.image_dst_scaled(),
        )
        return loss

    def loss__recon_l1_smooth(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        loss = F.smooth_l1_loss(
            pipeline.get_predicted_image(),
            pipeline.image_dst_normalized(),
            beta=0.001,
            reduction="mean",
        )
        return loss

    def loss__recon_l1_smooth_lab(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        loss = F.smooth_l1_loss(
            pipeline.get_predicted_image_lab(),
            pipeline.image_dst_lab(),
            beta=0.1,
            reduction="mean",
        )
        return loss

    def loss__recon_mse_lab(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        loss = F.mse_loss(
            pipeline.get_predicted_image_lab(),
            pipeline.image_dst_lab(),
            reduction="mean",
        )
        return loss

    def loss__recon_l1(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        loss = F.l1_loss(
            pipeline.get_predicted_image(),
            pipeline.image_dst_normalized(),
            reduction="mean",
        )
        return loss

    def loss__recon_mse(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        loss = F.mse_loss(
            pipeline.get_predicted_image(),
            pipeline.image_dst_normalized(),
            reduction="mean",
        )
        return loss

    def loss__psnr(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        mse = torch.mean(
            (
                pipeline.get_predicted_image_destandardized()
                - pipeline.image_dst_scaled()
            )
            ** 2,
            dim=(1, 2, 3),
        )
        psnr = 10 * torch.log10(1.0 / mse).mean()
        return psnr
