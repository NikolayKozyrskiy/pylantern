from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F

if TYPE_CHECKING:
    from ..pipeline import DeepfakePipeline


@torch.no_grad()
def original_image_src_base(pipeline: "DeepfakePipeline"):
    return pipeline.image_src_scaled().detach().cpu().clone()


@torch.no_grad()
def original_image_dst_base(pipeline: "DeepfakePipeline"):
    return pipeline.image_dst_scaled().detach().cpu().clone()


@torch.no_grad()
def masked_image_src_base(pipeline: "DeepfakePipeline"):
    return pipeline.masked_image_src().detach().cpu().clone()


@torch.no_grad()
def masked_image_dst_base(pipeline: "DeepfakePipeline"):
    return pipeline.masked_image_dst().detach().cpu().clone()


@torch.no_grad()
def mask_dst(pipeline: "DeepfakePipeline"):
    return (
        pipeline.segmentation_mask_dst()
        .detach()
        .cpu()
        .clone()
        .expand_as(pipeline.face_swapper_generate_image().detach().cpu().clone())
    )


@torch.no_grad()
def pred_image_base(pipeline: "DeepfakePipeline"):
    return pipeline.face_swapper_generate_image_destandardized().detach().cpu().clone()


@torch.no_grad()
def pred_image_base_masked(pipeline: "DeepfakePipeline"):
    return pipeline.face_swapper_generate_image_destandardized().detach().cpu().clone()


@torch.no_grad()
def pred_mask(pipeline: "DeepfakePipeline"):
    img = pipeline.segmentation_mask_predict_sigmoid().detach().cpu().clone()
    img = torch.clip(img, min=0.0, max=1.0).expand_as(
        pipeline.face_swapper_generate_image().detach().cpu().clone()
    )
    return img


@torch.no_grad()
def eyes_roi_gt_pred_x2(pipeline: "DeepfakePipeline"):
    left_eye_dst = torch.clip(
        pipeline.left_eye_roi_image_dst_destandardized().detach().cpu().clone(),
        min=0.0,
        max=1.0,
    )
    right_eye_dst = torch.clip(
        pipeline.right_eye_roi_image_dst_destandardized().detach().cpu().clone(),
        min=0.0,
        max=1.0,
    )
    left_eye_pred = torch.clip(
        pipeline.left_eye_roi_image_generated_destandardized().detach().cpu().clone(),
        min=0.0,
        max=1.0,
    )
    right_eye_pred = torch.clip(
        pipeline.right_eye_roi_image_generated_destandardized().detach().cpu().clone(),
        min=0.0,
        max=1.0,
    )
    eyes_gt = torch.concat((left_eye_dst, right_eye_dst), dim=3)
    eyes_pred = torch.concat((left_eye_pred, right_eye_pred), dim=3)
    eyes = torch.concat((eyes_gt, eyes_pred), dim=2)
    eyes = F.interpolate(eyes, scale_factor=2.0, mode="bilinear")
    h, w = eyes.shape[-2:]
    res = torch.zeros_like(pipeline.image_dst().detach().cpu(), dtype=torch.float32)
    res[:, :, :h, :w] = eyes
    return res


@torch.no_grad()
def pred_img_arcface_aligned(pipeline: "DeepfakePipeline"):
    img = (
        pipeline.generated_img_arcface_aligned_destandardized_224()
        .detach()
        .cpu()
        .clone()
    )
    img = F.interpolate(img, size=pipeline.image_dst_hw(), mode="bilinear")
    return img


@torch.no_grad()
def dst_img_arcface_aligned(pipeline: "DeepfakePipeline"):
    img = pipeline.dst_img_arcface_aligned_scaled_224().detach().cpu().clone()
    img = F.interpolate(img, size=pipeline.image_dst_hw(), mode="bilinear")
    return img
