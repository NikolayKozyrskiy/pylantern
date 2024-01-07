import math
from typing import TYPE_CHECKING, Optional, Union

import torch
from matches.shortcuts.dag import graph_node
from torch import Tensor, nn
from torch.nn import functional as F
from typing_extensions import override

from pylantern.common.utils import to_device

from .. import BasePix2PixPipeline

if TYPE_CHECKING:
    from pylantern.model_zoo.gfpgan import StyleGAN2Discriminator
    from pylantern.tasks.gan.pix2pix.configs import GFPGANConfig


class GFPGANPipeline(BasePix2PixPipeline):
    def __init__(
        self,
        config: "GFPGANConfig",
        generator_model: "nn.Module",
        discriminator_model: "StyleGAN2Discriminator",
        device: Union[str, torch.device],
    ):
        BasePix2PixPipeline.__init__(
            self,
            config=config,
            generator_model=generator_model,
            discriminator_model=discriminator_model,
            device=device,
        )
        self.config: "GFPGANConfig"
        self.discriminator_model: "StyleGAN2Discriminator"
        self.log_size = int(math.log(self.config.image_size[0], 2))
        self.use_rgbs = (
            "loss__pyramid_reconstruction"
            in self.config.criterion_aggregation.unique_loss_fn_names
        )

    @graph_node
    @override
    def segmentation_mask_predict(self):
        return self.segmentation_mask_and_pyramid_predict()[0]

    @graph_node
    def dst_img_normalized_pyramid(self):
        """Construct image pyramid for intermediate restoration loss"""
        pyramid_gt = [self.image_dst_normalized()]
        down_img = self.image_dst_normalized()
        for _ in range(0, self.log_size - 3):
            down_img = F.interpolate(
                down_img,
                scale_factor=0.5,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            pyramid_gt.insert(0, down_img)
        return pyramid_gt

    @graph_node
    def dst_segmentation_mask_pyramid(self):
        """Construct segmentation mask pyramid for intermediate restoration loss"""
        pyramid_gt = [self.segmentation_mask_dst()]
        down_mask = self.segmentation_mask_dst()
        for _ in range(0, self.log_size - 3):
            down_mask = F.interpolate(
                down_mask,
                scale_factor=0.5,
                mode="nearest",
                align_corners=False,
            )
            pyramid_gt.insert(0, down_mask)
        return pyramid_gt

    @graph_node
    def gfpgan_generator_forward(self):
        if self.use_rgbs:
            img, rgbs = self.generator_model(self.image_src(), return_rgb=True)
            return img, rgbs
        else:
            img, _ = self.generator_model(self.image_src(), return_rgb=False)
            return img, None

    @graph_node
    def segmentation_mask_and_pyramid_predict(self) -> Tensor:
        img, rgbs = self.gfpgan_generator_forward()
        if self.use_rgbs:
            rgb_masks = []
            for rgb in rgbs:
                rgb_masks.append(rgb[:, 3:, ...])
        else:
            rgb_masks = None
        return img[:, 3:, ...], rgb_masks

    @graph_node
    def segmentation_mask_and_pyramid_predict_sigmoid(self) -> Tensor:
        mask, rgb_masks = self.segmentation_mask_and_pyramid_predict()
        if self.use_rgbs:
            for i in range(len(rgb_masks)):
                rgb_masks[i] = torch.sigmoid(rgb_masks[i])
        return torch.sigmoid(mask), rgb_masks

    @graph_node
    def generate_image_and_pyramid_masked(self) -> Tensor:
        img, rgbs = self.gfpgan_generator_forward()
        mask, rgb_masks = self.segmentation_mask_and_pyramid_predict_sigmoid()
        img_blended = img[:, :3, ...] * mask + (1 - mask) * self.image_src_normalized()
        if self.use_rgbs:
            rgbs_blended = []
            for rgb, rgb_mask in zip(rgbs, rgb_masks):
                src_img = F.interpolate(
                    input=self.image_src_normalized(),
                    size=[rgb.shape[2], rgb.shape[3]],
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                rgbs_blended.append(
                    rgb[:, :3, ...] * rgb_mask + (1 - rgb_mask) * src_img
                )
        else:
            rgbs_blended = None
        return img_blended, rgbs_blended

    @graph_node
    @override
    def get_predicted_image(self) -> Tensor:
        pred = (
            self.generate_image_and_pyramid_masked()[0]
            if self.config.predict_mask
            else self.gfpgan_generator_forward()[0]
        )
        return pred

    @graph_node
    def predicted_rgbs_normalized_pyramid(self):
        pyramid = self.generate_image_and_pyramid_masked()[1]
        return pyramid

    @graph_node
    def discriminate_generator_fake_normalized_input(self) -> Tensor:
        generator_output = self.get_predicted_image()
        discriminator_output = self.discriminator_model(generator_output)
        return discriminator_output

    @graph_node
    def discriminate_discriminator_fake_normalized_input(self) -> Tensor:
        generator_output = self.get_predicted_image().detach()
        discriminator_output = self.discriminator_model(generator_output)
        return discriminator_output

    @graph_node
    def discriminate_discriminator_dst_normalized_input(self) -> Tensor:
        dst_img = self.image_dst_normalized()
        discriminator_output = self.discriminator_model(dst_img)
        return discriminator_output

    @graph_node
    def discriminate_discriminator_dst_normalized_with_grad(self):
        dst_img = self.image_dst_normalized_with_grad()
        real_pred = self.discriminator_model(dst_img)
        return real_pred


def gfpgan_pipeline_from_config(
    config: "GFPGANConfig",
    device: Union[str, torch.device],
) -> GFPGANPipeline:
    return GFPGANPipeline(
        config=config,
        generator_model=to_device(config.generator_model(), device=device),
        discriminator_model=to_device(config.discriminator_model(), device=device),
        device=device,
    )
