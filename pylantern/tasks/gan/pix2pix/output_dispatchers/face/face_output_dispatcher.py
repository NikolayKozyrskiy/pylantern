from typing import TYPE_CHECKING, List

import torch
import torch.nn.functional as F
from matches.loop import Loop

from pylantern.common.nn.functional import gram_matrix, mse_reduction
from pylantern.tasks.gan.pix2pix.criterions.face import (
    face_complex_criterions as face_cc,
)
from pylantern.tasks.gan.pix2pix.output_dispatchers.general.gfpgan_output_dispatcher import (
    GFPGANOutputDispatcher,
)
from pylantern.tasks.gan.pix2pix.output_dispatchers.general.pix2pixhd_output_dispatcher import (
    Pix2PixHDOutputDispatcher,
)
from pylantern.tasks.gan.pix2pix.output_dispatchers.output_dispatcher import (
    BasePix2PixOutputDispatcher,
)

if TYPE_CHECKING:
    from pylantern.tasks.gan.pix2pix.configs import (
        FaceGFPGANConfig,
        FacePix2PixConfig,
        FacePix2PixHDConfig,
        FaceSpadeConfig,
    )
    from pylantern.tasks.gan.pix2pix.pipelines import FacePix2PixPipeline


class FacePix2PixOutputDispatcher(BasePix2PixOutputDispatcher):
    def __init__(self, config: "FacePix2PixConfig", *args, **kwargs):
        BasePix2PixOutputDispatcher.__init__(self, config=config, *args, **kwargs)
        self._prepare_complex_criterions(
            complex_criterions_module=face_cc, *args, **kwargs
        )

    def loss__identity_arcface_resnet18(
        self, pipeline: "FacePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        identity_gt = self.complex_criterions.gfpgan_identity_arcface_resnet18(
            pipeline.dst_img_normalized_gray_128()
        ).detach()
        identity_out = self.complex_criterions.gfpgan_identity_arcface_resnet18(
            pipeline.get_predicted_image_normalized_gray_128()
        )
        loss = F.smooth_l1_loss(identity_out, identity_gt, beta=0.01, reduction="mean")
        return loss

    def loss__identity_arcface_iresnet18(
        self, pipeline: "FacePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        identity_gt = self.complex_criterions.identity_arcface_iresnet18(
            pipeline.image_dst_normalized_112()
        ).detach()
        identity_out = self.complex_criterions.identity_arcface_iresnet18(
            pipeline.get_predicted_image_normalized_112()
        )
        loss = F.smooth_l1_loss(identity_out, identity_gt, beta=0.01, reduction="mean")
        return loss

    def loss__identity_arcface_iresnet34(
        self, pipeline: "FacePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        identity_gt = self.complex_criterions.identity_arcface_iresnet34(
            pipeline.image_dst_normalized_112()
        ).detach()
        identity_out = self.complex_criterions.identity_arcface_iresnet34(
            pipeline.get_predicted_image_normalized_112()
        )
        loss = F.smooth_l1_loss(identity_out, identity_gt, beta=0.01, reduction="mean")
        return loss

    def loss__identity_arcface_iresnet50(
        self, pipeline: "FacePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        identity_gt = self.complex_criterions.identity_arcface_iresnet50(
            pipeline.image_dst_normalized_112()
        ).detach()
        identity_out = self.complex_criterions.identity_arcface_iresnet50(
            pipeline.get_predicted_image_normalized_112()
        )
        loss = F.smooth_l1_loss(identity_out, identity_gt, beta=0.01, reduction="mean")
        return loss

    def loss__identity_arcface_iresnet100(
        self, pipeline: "FacePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        identity_gt = self.complex_criterions.identity_arcface_iresnet100(
            pipeline.image_dst_normalized_112()
        ).detach()
        identity_out = self.complex_criterions.identity_arcface_iresnet100(
            pipeline.get_predicted_image_normalized_112()
        )
        loss = F.smooth_l1_loss(identity_out, identity_gt, beta=0.01, reduction="mean")
        return loss

    def loss__identity_face_clip(
        self, pipeline: "FacePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        identity_gt = self.complex_criterions.face_clip.encode_image(
            pipeline.image_dst_resized_scaled_224()
            # pipeline.dst_img_arcface_aligned_scaled_224()
        ).detach()
        identity_out = self.complex_criterions.face_clip.encode_image(
            pipeline.get_predicted_image_resized_destandardized_224()
            # pipeline.generated_img_arcface_aligned_destandardized_224()
        )
        loss = F.smooth_l1_loss(identity_out, identity_gt, beta=0.01, reduction="mean")
        return loss

    def component__generator_left_eye(
        self, pipeline: "FacePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        return self.complex_criterions.components_gan_loss(
            pipeline.discriminator_left_eye_fake()[0],
            target_is_real=True,
            is_disc=False,
        )

    def component__generator_right_eye(
        self, pipeline: "FacePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        return self.complex_criterions.components_gan_loss(
            pipeline.discriminator_right_eye_fake()[0],
            target_is_real=True,
            is_disc=False,
        )

    def component__generator_mouth(
        self, pipeline: "FacePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        return self.complex_criterions.components_gan_loss(
            pipeline.discriminator_mouth_fake()[0], target_is_real=True, is_disc=False
        )

    def component__style_left_eye(
        self, pipeline: "FacePix2PixPipeline", loop: "Loop", *args, **kwargs
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
        self, pipeline: "FacePix2PixPipeline", loop: "Loop", *args, **kwargs
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
        self, pipeline: "FacePix2PixPipeline", loop: "Loop", *args, **kwargs
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
        self, pipeline: "FacePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        fake_d_pred = pipeline.discriminate_left_eye_fake()
        real_d_pred = pipeline.discriminate_left_eye_dst()
        loss = self.complex_criterions.components_gan_loss(
            real_d_pred, target_is_real=True, is_disc=True
        )
        loss = loss + mse_reduction(fake_d_pred, 0.0, reduction="mean")
        return loss

    def component__disc_right_eye(
        self, pipeline: "FacePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        fake_d_pred = pipeline.discriminate_right_eye_fake()
        real_d_pred = pipeline.discriminate_right_eye_dst()
        loss = self.complex_criterions.components_gan_loss(
            real_d_pred, target_is_real=True, is_disc=True
        )
        loss = loss + mse_reduction(fake_d_pred, 0.0, reduction="mean")
        return loss

    def component__disc_mouth(
        self, pipeline: "FacePix2PixPipeline", loop: "Loop", *args, **kwargs
    ):
        fake_d_pred = pipeline.discriminate_mouth_fake()
        real_d_pred = pipeline.discriminate_mouth_dst()
        loss = self.complex_criterions.components_gan_loss(
            real_d_pred, target_is_real=True, is_disc=True
        )
        loss = loss + mse_reduction(fake_d_pred, 0.0, reduction="mean")
        return loss


class FaceGFPGANOutputDispatcher(FacePix2PixOutputDispatcher, GFPGANOutputDispatcher):
    def __init__(self, config: "FaceGFPGANConfig", *args, **kwargs):
        FacePix2PixOutputDispatcher.__init__(self, config=config, *args, **kwargs)


class FacePix2PixHDOutputDispatcher(
    FacePix2PixOutputDispatcher, Pix2PixHDOutputDispatcher
):
    def __init__(self, config: "FacePix2PixHDConfig", *args, **kwargs):
        FacePix2PixOutputDispatcher.__init__(self, config=config, *args, **kwargs)


class FaceSpadeOutputDispatcher(FacePix2PixOutputDispatcher, Pix2PixHDOutputDispatcher):
    def __init__(self, config: "FaceSpadeConfig", *args, **kwargs):
        FacePix2PixOutputDispatcher.__init__(self, config=config, *args, **kwargs)
