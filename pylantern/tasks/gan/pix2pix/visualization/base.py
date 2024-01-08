from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pylantern.tasks.gan.pix2pix.pipelines import BasePix2PixPipeline


@torch.no_grad()
def original_image_src_base(pipeline: "BasePix2PixPipeline"):
    return pipeline.image_src_scaled().detach().cpu().clone()


@torch.no_grad()
def original_image_dst_base(pipeline: "BasePix2PixPipeline"):
    return pipeline.image_dst_scaled().detach().cpu().clone()


@torch.no_grad()
def masked_image_src_base(pipeline: "BasePix2PixPipeline"):
    return pipeline.masked_image_src().detach().cpu().clone()


@torch.no_grad()
def masked_image_dst_base(pipeline: "BasePix2PixPipeline"):
    return pipeline.masked_image_dst().detach().cpu().clone()


@torch.no_grad()
def mask_dst(pipeline: "BasePix2PixPipeline"):
    return (
        pipeline.segmentation_mask_dst()
        .detach()
        .cpu()
        .clone()
        .expand_as(pipeline.get_predicted_image().detach().cpu().clone())
    )


@torch.no_grad()
def pred_image_base(pipeline: "BasePix2PixPipeline"):
    return pipeline.get_predicted_image_destandardized().detach().cpu().clone()


@torch.no_grad()
def pred_image_base_masked(pipeline: "BasePix2PixPipeline"):
    return pipeline.get_predicted_image_destandardized().detach().cpu().clone()


@torch.no_grad()
def pred_mask(pipeline: "BasePix2PixPipeline"):
    img = pipeline.segmentation_mask_predict_sigmoid().detach().cpu().clone()
    img = torch.clip(img, min=0.0, max=1.0).expand_as(
        pipeline.get_predicted_image().detach().cpu().clone()
    )
    return img
