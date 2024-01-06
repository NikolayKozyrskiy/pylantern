from typing import TYPE_CHECKING, List

import torch
import torch.nn.functional as F
from basicsr.losses.gan_loss import r1_penalty
from matches.loop import Loop

from pylantern.common.nn.functional import gram_matrix, mse_reduction
from pylantern.output_dispatcher import BaseOutputDispatcher
from pylantern.tasks.gan.common.nn.gan_loss import compute_pix2pix_hd_mse

from . import complex_criterions as cc

if TYPE_CHECKING:
    from .configs import (
        BasePix2PixConfig,
        GanPix2PixConfig,
        GFPGANConfig,
        Pix2PixHDConfig,
    )
    from .pipelines import (
        BasePix2PixPipeline,
        GanPix2PixPipeline,
        GFPGANPipeline,
        Pix2PixHDPipeline,
    )


class BasePix2PixOutputDispatcher(BaseOutputDispatcher):
    def __init__(self, config: "BasePix2PixConfig", *args, **kwargs):
        super().__init__(config=config, complex_criterions_module=cc, *args, **kwargs)

    def gan__temporal_mse(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        cropped_orig_predict = pipeline.random_cropped_orig_0875_predict()
        cropped_predict = pipeline.random_cropped_0875_predict()
        coeff = int(loop.iterations.current_epoch) / pipeline.config.max_epoch
        coeff = coeff ** (1.0 / 4)
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
        self, pipeline: "GanPix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        loss = self.complex_criterions.vision_aided_discriminator(
            pipeline.image_dst_normalized(), for_real=True
        ).mean()
        loss = (
            loss
            + self.complex_criterions.vision_aided_discriminator(
                pipeline.face_swapper_generate_image(), for_real=False
            ).mean()
        )
        return loss

    def gan__vision_aided_generator(
        self, pipeline: "GanPix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        loss = self.complex_criterions.vision_aided_discriminator(
            pipeline.face_swapper_generate_image(), for_G=True
        ).mean()
        return loss

    def loss__vgg_perceptual(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        loss = (
            self.complex_criterions.vgg19_perceptual(
                pipeline.face_swapper_generate_image_destandardized(),
                pipeline.image_dst_scaled().detach(),
            )
            * pipeline.config.lambda_feat
        )
        return loss

    def loss__vgg19_perceptual_and_style(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        loss = self.complex_criterions.vgg19_perceptual_and_style(
            pipeline.face_swapper_generate_image_destandardized(),
            pipeline.image_dst_scaled().detach(),
        )
        return loss

    def loss__lpips_perceptual(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        loss = self.complex_criterions.lpips_alex_perceptual(
            pipeline.face_swapper_generate_image(),
            pipeline.image_dst_normalized(),
        ).mean()
        return loss

    def loss__dreamsim_cos_ensemble_perceptual(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        pred = self.complex_criterions.dreamsim_ensemble.embed(
            pipeline.generated_img_resized_destandardized_224()
        )
        gt = self.complex_criterions.dreamsim_ensemble.embed(
            pipeline.dst_img_resized_scaled_224(),
        ).detach()
        return (1 - F.cosine_similarity(pred, gt, dim=-1)).mean()

    def loss__identity_arcface_resnet18(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        identity_gt = self.complex_criterions.gfpgan_identity_arcface_resnet18(
            pipeline.dst_img_normalized_gray_128()
        ).detach()
        identity_out = self.complex_criterions.gfpgan_identity_arcface_resnet18(
            pipeline.generated_img_normalized_gray_128()
        )
        loss = F.smooth_l1_loss(identity_out, identity_gt, beta=0.01, reduction="mean")
        return loss

    def loss__identity_arcface_iresnet18(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        identity_gt = self.complex_criterions.identity_arcface_iresnet18(
            pipeline.dst_img_normalized_112()
        ).detach()
        identity_out = self.complex_criterions.identity_arcface_iresnet18(
            pipeline.generated_img_normalized_112()
        )
        loss = F.smooth_l1_loss(identity_out, identity_gt, beta=0.01, reduction="mean")
        return loss

    def loss__identity_arcface_iresnet34(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        identity_gt = self.complex_criterions.identity_arcface_iresnet34(
            pipeline.dst_img_normalized_112()
        ).detach()
        identity_out = self.complex_criterions.identity_arcface_iresnet34(
            pipeline.generated_img_normalized_112()
        )
        loss = F.smooth_l1_loss(identity_out, identity_gt, beta=0.01, reduction="mean")
        return loss

    def loss__identity_arcface_iresnet50(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        identity_gt = self.complex_criterions.identity_arcface_iresnet50(
            pipeline.dst_img_normalized_112()
        ).detach()
        identity_out = self.complex_criterions.identity_arcface_iresnet50(
            pipeline.generated_img_normalized_112()
        )
        loss = F.smooth_l1_loss(identity_out, identity_gt, beta=0.01, reduction="mean")
        return loss

    def loss__identity_arcface_iresnet100(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        identity_gt = self.complex_criterions.identity_arcface_iresnet100(
            pipeline.dst_img_normalized_112()
        ).detach()
        identity_out = self.complex_criterions.identity_arcface_iresnet100(
            pipeline.generated_img_normalized_112()
        )
        loss = F.smooth_l1_loss(identity_out, identity_gt, beta=0.01, reduction="mean")
        return loss

    def loss__identity_face_clip(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        identity_gt = self.complex_criterions.face_clip.encode_image(
            pipeline.dst_img_resized_scaled_224()
            # pipeline.dst_img_arcface_aligned_scaled_224()
        ).detach()
        identity_out = self.complex_criterions.face_clip.encode_image(
            pipeline.generated_img_resized_destandardized_224()
            # pipeline.generated_img_arcface_aligned_destandardized_224()
        )
        loss = F.smooth_l1_loss(identity_out, identity_gt, beta=0.01, reduction="mean")
        return loss

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
            pipeline.face_swapper_generate_image_destandardized(),
            pipeline.image_dst_scaled(),
        )
        return loss

    def loss__recon_ms_dssim(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        loss = self.complex_criterions.ms_dssim(
            pipeline.face_swapper_generate_image_destandardized(),
            pipeline.image_dst_scaled(),
        )
        return loss

    def loss__recon_l1_smooth(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        loss = F.smooth_l1_loss(
            pipeline.face_swapper_generate_image(),
            pipeline.image_dst_normalized(),
            beta=0.001,
            reduction="mean",
        )
        return loss

    def loss__recon_l1_smooth_lab(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        loss = F.smooth_l1_loss(
            pipeline.face_swapper_generate_image_lab(),
            pipeline.image_dst_lab(),
            beta=0.1,
            reduction="mean",
        )
        return loss

    def loss__recon_mse_lab(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        loss = F.mse_loss(
            pipeline.face_swapper_generate_image_lab(),
            pipeline.image_dst_lab(),
            reduction="mean",
        )
        return loss

    def loss__recon_l1(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        loss = F.l1_loss(
            pipeline.face_swapper_generate_image(),
            pipeline.image_dst_normalized(),
            reduction="mean",
        )
        return loss

    def loss__recon_mse(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        loss = F.mse_loss(
            pipeline.face_swapper_generate_image(),
            pipeline.image_dst_normalized(),
            reduction="mean",
        )
        return loss

    def loss__psnr(
        self, pipeline: "BasePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        mse = torch.mean(
            (
                pipeline.face_swapper_generate_image_destandardized()
                - pipeline.image_dst_scaled()
            )
            ** 2,
            dim=(1, 2, 3),
        )
        psnr = 10 * torch.log10(1.0 / mse).mean()
        return psnr


class GanPix2PixOutputDispatcher(BasePix2PixOutputDispatcher):
    def __init__(self, config: "GanPix2PixConfig", *args, **kwargs):
        super().__init__(config=config, *args, **kwargs)

    def component__generator_left_eye(
        self, pipeline: "GanPix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        return self.complex_criterions.components_gan_loss(
            pipeline.discriminator_left_eye_fake()[0],
            target_is_real=True,
            is_disc=False,
        )

    def component__generator_right_eye(
        self, pipeline: "GanPix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        return self.complex_criterions.components_gan_loss(
            pipeline.discriminator_right_eye_fake()[0],
            target_is_real=True,
            is_disc=False,
        )

    def component__generator_mouth(
        self, pipeline: "GanPix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        return self.complex_criterions.components_gan_loss(
            pipeline.discriminator_mouth_fake()[0], target_is_real=True, is_disc=False
        )

    def component__style_left_eye(
        self, pipeline: "GanPix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        fake_feats = pipeline.discriminator_left_eye_fake()[1]
        real_feats = pipeline.discriminator_left_eye_dst()[1]
        loss = F.l1_loss(
            gram_matrix(fake_feats[0]),
            gram_matrix(real_feats[0].detach()),
            reduction="mean",
        ) * 0.5 + F.l1_loss(
            gram_matrix(fake_feats[1]),
            gram_matrix(real_feats[1].detach()),
            reduction="mean",
        )
        return loss

    def component__style_right_eye(
        self, pipeline: "GanPix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        fake_feats = pipeline.discriminator_right_eye_fake()[1]
        real_feats = pipeline.discriminator_right_eye_dst()[1]
        loss = F.l1_loss(
            gram_matrix(fake_feats[0]),
            gram_matrix(real_feats[0].detach()),
            reduction="mean",
        ) * 0.5 + F.l1_loss(
            gram_matrix(fake_feats[1]),
            gram_matrix(real_feats[1].detach()),
            reduction="mean",
        )
        return loss

    def component__style_mouth(
        self, pipeline: "GanPix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        fake_feats = pipeline.discriminator_mouth_fake()[1]
        real_feats = pipeline.discriminator_mouth_dst()[1]
        loss = F.l1_loss(
            gram_matrix(fake_feats[0]),
            gram_matrix(real_feats[0].detach()),
            reduction="mean",
        ) * 0.5 + F.l1_loss(
            gram_matrix(fake_feats[1]),
            gram_matrix(real_feats[1].detach()),
            reduction="mean",
        )
        return loss

    def component__disc_left_eye(
        self, pipeline: "GanPix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        fake_d_pred = pipeline.discriminate_left_eye_fake()
        real_d_pred = pipeline.discriminate_left_eye_dst()
        loss = self.complex_criterions.components_gan_loss(
            real_d_pred, target_is_real=True, is_disc=True
        )
        loss = loss + mse_reduction(fake_d_pred, 0.0, reduction="mean")
        return loss

    def component__disc_right_eye(
        self, pipeline: "GanPix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        fake_d_pred = pipeline.discriminate_right_eye_fake()
        real_d_pred = pipeline.discriminate_right_eye_dst()
        loss = self.complex_criterions.components_gan_loss(
            real_d_pred, target_is_real=True, is_disc=True
        )
        loss = loss + mse_reduction(fake_d_pred, 0.0, reduction="mean")
        return loss

    def component__disc_mouth(
        self, pipeline: "GanPix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        fake_d_pred = pipeline.discriminate_mouth_fake()
        real_d_pred = pipeline.discriminate_mouth_dst()
        loss = self.complex_criterions.components_gan_loss(
            real_d_pred, target_is_real=True, is_disc=True
        )
        loss = loss + mse_reduction(fake_d_pred, 0.0, reduction="mean")
        return loss


class Pix2PixHDOutputDispatcher(GanPix2PixOutputDispatcher):
    def __init__(self, config: "Pix2PixHDConfig", *args, **kwargs):
        super().__init__(config=config, *args, **kwargs)

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


class GFPGANOutputDispatcher(GanPix2PixOutputDispatcher):
    def __init__(self, config: "GFPGANConfig", *args, **kwargs):
        super().__init__(config=config, *args, **kwargs)

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
            x=pipeline.face_swapper_generate_image(),
            gt=pipeline.image_dst_normalized(),
        )
        return loss + loss_style
