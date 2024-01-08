from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F

if TYPE_CHECKING:
    from pylantern.tasks.gan.pix2pix.pipelines import FacePix2PixPipeline


@torch.no_grad()
def eyes_roi_gt_pred_x2(pipeline: "FacePix2PixPipeline"):
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
def pred_img_arcface_aligned(pipeline: "FacePix2PixPipeline"):
    img = (
        pipeline.get_predicted_image_arcface_aligned_destandardized_224()
        .detach()
        .cpu()
        .clone()
    )
    img = F.interpolate(img, size=pipeline.image_dst_hw(), mode="bilinear")
    return img


@torch.no_grad()
def dst_img_arcface_aligned(pipeline: "FacePix2PixPipeline"):
    img = pipeline.dst_img_arcface_aligned_scaled_224().detach().cpu().clone()
    img = F.interpolate(img, size=pipeline.image_dst_hw(), mode="bilinear")
    return img
