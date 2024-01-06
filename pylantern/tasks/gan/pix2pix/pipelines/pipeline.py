from contextlib import contextmanager
from functools import cached_property
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
from kornia.color import rgb_to_lab
from kornia.geometry.transform import warp_affine
from matches.shortcuts.dag import ComputationGraph, graph_node
from torch import Tensor, nn
from torchvision.ops import roi_align

from pylantern.common.constants import ARCFACE_DST_KPS
from pylantern.common.nn.functional import atm_arcface, rgb2gray
from pylantern.common.utils import to_device
from pylantern.common.utils.img import (
    crop_img_tensor_by_params,
    crop_random_params,
    interpolate_img_tensor,
)
from pylantern.common.utils.tensor import denormalize_tensor, normalize_tensor

if TYPE_CHECKING:
    from pylantern.model_zoo.gfpgan import FacialComponentDiscriminator

    from ..configs import BasePix2PixConfig, GanPix2PixConfig


class BasePix2PixPipeline(ComputationGraph):
    def __init__(
        self,
        config: "BasePix2PixConfig",
        generator_model: "nn.Module",
        device: Union[str, torch.device],
    ):
        super().__init__()
        self.config = config
        self.generator_model = generator_model
        self.device = device
        self._mean = self.generator_model.mean
        self._std = self.generator_model.std
        self._max_orig_val = torch.tensor(255.0, dtype=torch.float32).to(device)
        self._arcface_dst_kps = (
            torch.tensor(ARCFACE_DST_KPS, dtype=torch.float64).to(device)
            if any(
                k in config.attrs_to_load.keys()
                for k in ("kps_5_buffalo_l_dst", "kps_5_buffalo_l_src")
            )
            else None
        )
        self.batch = None

    @contextmanager
    def batch_scope(self, batch: Dict[str, torch.Tensor]):
        try:
            with self.cache_scope():
                self.batch = batch
                yield
        finally:
            self.batch = None

    @cached_property
    def eye_roi_size(self) -> int:
        return int(round(self.config.eye_roi_relative_size * self.config.image_size[0]))

    @cached_property
    def mouth_roi_size(self) -> int:
        return int(
            round(self.config.mouth_roi_relative_size * self.config.image_size[0])
        )

    @graph_node
    def image_src(self) -> Tensor:
        return self.batch["image_src"]

    @graph_node
    def image_src_hw(self) -> Tuple[int, int]:
        image = self.image_src()
        h, w = image.shape[2:]
        return h, w

    @graph_node
    @torch.no_grad()
    def image_src_normalized(self) -> Tensor:
        img = self.image_src() / self._max_orig_val
        res = img - self._mean.expand_as(img)
        res = res / self._std.expand_as(res)
        return res

    @graph_node
    @torch.no_grad()
    def image_src_scaled(self) -> Tensor:
        img = self.image_src() / self._max_orig_val
        return img

    @graph_node
    @torch.no_grad()
    def image_src_lab(self) -> Tensor:
        img_lab = rgb_to_lab(self.image_src_scaled())
        return img_lab

    @graph_node
    @torch.no_grad()
    def image_dst(self) -> Tensor:
        img = (
            self.image_dst_orig() * self.segmentation_mask_dst()
            + (1 - self.segmentation_mask_dst()) * self.image_src()
        )
        img = torch.clip(img, min=0.0, max=255.0)
        return img

    @graph_node
    def image_dst_orig(self) -> Tensor:
        return self.batch["image_dst"]

    @graph_node
    def image_dst_hw(self) -> Tuple[int, int]:
        image = self.image_dst()
        h, w = image.shape[2:]
        return h, w

    @graph_node
    @torch.no_grad()
    def image_dst_scaled(self) -> Tensor:
        img = self.image_dst() / self._max_orig_val
        return img

    @graph_node
    @torch.no_grad()
    def image_dst_normalized(self) -> Tensor:
        img = self.image_dst_scaled()
        res = img - self._mean.expand_as(img)
        res = res / self._std.expand_as(res)
        return res

    @graph_node
    def image_dst_normalized_with_grad(self) -> Tensor:
        img = self.image_dst_normalized().detach().clone()
        img.requires_grad = True
        return img

    @graph_node
    @torch.no_grad()
    def image_dst_lab(self) -> Tensor:
        dst_img_scaled = self.image_dst_scaled()
        img_lab = rgb_to_lab(dst_img_scaled)
        return img_lab

    @graph_node
    def segmentation_mask_src(self) -> Tensor:
        return self.batch["mask_src"]

    @graph_node
    def segmentation_mask_dst(self) -> Tensor:
        return self.batch["mask_blended_dst"]

    @graph_node
    def masked_image_src(self) -> Tensor:
        return self.image_src() * self.segmentation_mask_src()

    @graph_node
    def masked_image_dst(self) -> Tensor:
        return self.image_dst_normalized() * self.segmentation_mask_dst()

    @graph_node
    def kps_5_buffalo_l_dst(self) -> Tensor:
        return self.batch["kps_5_buffalo_l_dst"]

    @graph_node
    def kps_5_buffalo_l_src(self) -> Tensor:
        return self.batch["kps_5_buffalo_l_src"]

    @graph_node
    @torch.no_grad()
    def face_swapper_predict(self) -> Tensor:
        pred = self.generator_model.predict_image(self.image_src())
        return pred

    @graph_node
    def face_swapper_forward(self) -> Tensor:
        pred = self.generator_model(self.image_src())
        return pred

    @graph_node
    def segmentation_mask_predict(self) -> Tensor:
        return self.face_swapper_forward()[:, 3:, ...]

    @graph_node
    def segmentation_mask_predict_sigmoid(self) -> Tensor:
        return torch.sigmoid(self.segmentation_mask_predict())

    @graph_node
    def generate_image_masked(self) -> Tensor:
        img = self.face_swapper_forward()[:, :3, ...]
        mask = self.segmentation_mask_predict_sigmoid()
        img = img * mask + (1 - mask) * self.image_src_normalized()
        return img

    @graph_node
    def face_swapper_generate_image(self) -> Tensor:
        pred = (
            self.generate_image_masked()
            if self.config.predict_mask
            else self.face_swapper_forward()
        )
        return pred

    @graph_node
    def face_swapper_generate_image_destandardized(self) -> Tensor:
        img = self.face_swapper_generate_image()
        res = img * self._std.expand_as(img)
        res = res + self._mean.expand_as(res)
        return torch.clip(res, min=0.0, max=1.0)

    @graph_node
    def face_swapper_generate_image_lab(self) -> Tensor:
        img = self.face_swapper_generate_image_destandardized()
        img_lab = rgb_to_lab(img)
        return img_lab

    @graph_node
    def get_random_crop_params(self, crop_ratio: float) -> Tuple[int, int, int, int]:
        h, w = self.image_dst_hw()
        return crop_random_params(h_orig=h, w_orig=w, crop_ratio=crop_ratio)

    @graph_node
    def random_cropped_orig_0875_predict(self) -> Tensor:
        (
            h_crop_start,
            h_crop_len,
            w_crop_start,
            w_crop_len,
        ) = self.get_random_crop_params(crop_ratio=0.875)
        img_pred_cropped = crop_img_tensor_by_params(
            img=self.face_swapper_generate_image(),
            h_crop_start=h_crop_start,
            h_crop_len=h_crop_len,
            w_crop_start=w_crop_start,
            w_crop_len=w_crop_len,
        )
        return img_pred_cropped

    @graph_node
    def random_cropped_0875_predict(self) -> Tensor:
        (
            h_crop_start,
            h_crop_len,
            w_crop_start,
            w_crop_len,
        ) = self.get_random_crop_params(crop_ratio=0.875)
        img_cropped_input = crop_img_tensor_by_params(
            img=self.image_src(),
            h_crop_start=h_crop_start,
            h_crop_len=h_crop_len,
            w_crop_start=w_crop_start,
            w_crop_len=w_crop_len,
        ).detach()
        pred = self.generator_model(img_cropped_input)
        img = pred[:, :3, ...]
        mask = torch.sigmoid(pred[:, 3:, ...])
        img_cropped_input_normalized = normalize_tensor(
            tensor=img_cropped_input,
            scale=self._max_orig_val,
            mean=self._mean,
            std=self._std,
        )
        img_cropped_pred = img * mask + (1 - mask) * img_cropped_input_normalized
        return img_cropped_pred

    @graph_node
    def dst_img_normalized_gray_128(self) -> Tensor:
        out_gray = rgb2gray(self.image_dst_normalized(), keepdim=True)
        return interpolate_img_tensor(img=out_gray, size=(128, 128))

    @graph_node
    def generated_img_normalized_gray_128(self) -> Tensor:
        out_gray = rgb2gray(self.face_swapper_generate_image(), keepdim=True)
        return interpolate_img_tensor(img=out_gray, size=(128, 128))

    @graph_node
    def dst_img_normalized_112(self) -> Tensor:
        return interpolate_img_tensor(img=self.image_dst_normalized(), size=(112, 112))

    @graph_node
    def generated_img_normalized_112(self) -> Tensor:
        return interpolate_img_tensor(
            img=self.face_swapper_generate_image(), size=(112, 112)
        )

    @graph_node
    def dst_img_resized_scaled_224(self) -> Tensor:
        return interpolate_img_tensor(img=self.image_dst_scaled(), size=(224, 224))

    @graph_node
    def generated_img_resized_destandardized_224(self) -> Tensor:
        return interpolate_img_tensor(
            img=self.face_swapper_generate_image_destandardized(), size=(224, 224)
        )

    @graph_node
    def affine_transform_matrix_arcface_dst(self, image_size: int) -> Tensor:
        atm = atm_arcface(
            src_points=self.kps_5_buffalo_l_dst(),
            dst_points=self._arcface_dst_kps,
            image_size=image_size,
        )
        return atm

    @graph_node
    def dst_img_arcface_aligned_normalized_112(self) -> Tensor:
        img = self.image_dst_normalized()
        atm = self.affine_transform_matrix_arcface_dst(image_size=112)
        img_aligned = warp_affine(src=img, M=atm, dsize=(112, 112))
        return img_aligned

    @graph_node
    def generated_img_arcface_aligned_normalized_112(self) -> Tensor:
        img = self.face_swapper_generate_image()
        atm = self.affine_transform_matrix_arcface_dst(image_size=112)
        img_aligned = warp_affine(src=img, M=atm, dsize=(112, 112))
        return img_aligned

    @graph_node
    def dst_img_arcface_aligned_scaled_224(self) -> Tensor:
        img = self.image_dst_scaled()
        atm = self.affine_transform_matrix_arcface_dst(image_size=224)
        img_aligned = warp_affine(src=img, M=atm, dsize=(224, 224))
        return img_aligned

    @graph_node
    def generated_img_arcface_aligned_destandardized_224(self) -> Tensor:
        img = self.face_swapper_generate_image_destandardized()
        atm = self.affine_transform_matrix_arcface_dst(image_size=224)
        img_aligned = warp_affine(src=img, M=atm, dsize=(224, 224))
        return img_aligned

    @graph_node
    def eyes_roi_bboxes(self) -> Tensor:
        eyes_roi_bboxes = []
        loc_left_eyes = self.batch["left_eye_bbox"]
        loc_right_eyes = self.batch["right_eye_bbox"]
        for b in range(loc_left_eyes.size(0)):
            bbox = torch.stack([loc_left_eyes[b, :], loc_right_eyes[b, :]], dim=0)
            eyes_roi_bboxes.append(
                torch.cat(
                    [
                        loc_left_eyes.new_full((2, 1), b),
                        bbox,
                    ],
                    dim=-1,
                )
            )
        eyes_roi_bboxes = torch.cat(eyes_roi_bboxes, dim=0).to(loc_left_eyes)
        return eyes_roi_bboxes

    @graph_node
    def eyes_roi_image_dst_normalized(self) -> Tuple[Tensor, Tensor]:
        eyes = roi_align(
            input=self.image_dst_normalized(),
            boxes=self.eyes_roi_bboxes(),
            output_size=self.eye_roi_size,
        )
        return eyes[0::2, :, :, :], eyes[1::2, :, :, :]

    @graph_node
    def left_eye_roi_image_dst_normalized(self) -> Tensor:
        return self.eyes_roi_image_dst_normalized()[0]

    @graph_node
    def right_eye_roi_image_dst_normalized(self) -> Tensor:
        return self.eyes_roi_image_dst_normalized()[1]

    @graph_node
    @torch.no_grad()
    def left_eye_roi_image_dst_destandardized(self) -> Tensor:
        img = self.left_eye_roi_image_dst_normalized()
        res = img * self._std.expand_as(img)
        res = res + self._mean.expand_as(res)
        return res

    @graph_node
    @torch.no_grad()
    def right_eye_roi_image_dst_destandardized(self) -> Tensor:
        img = self.right_eye_roi_image_dst_normalized()
        res = img * self._std.expand_as(img)
        res = res + self._mean.expand_as(res)
        return res

    @graph_node
    def eyes_roi_image_generated(self) -> Tuple[Tensor, Tensor]:
        eyes = roi_align(
            input=self.face_swapper_generate_image(),
            boxes=self.eyes_roi_bboxes(),
            output_size=self.eye_roi_size,
        )
        return eyes[0::2, :, :, :], eyes[1::2, :, :, :]

    @graph_node
    def left_eye_roi_image_generated(self) -> Tensor:
        return self.eyes_roi_image_generated()[0]

    @graph_node
    def right_eye_roi_image_generated(self) -> Tensor:
        return self.eyes_roi_image_generated()[1]

    @graph_node
    @torch.no_grad()
    def left_eye_roi_image_generated_destandardized(self) -> Tensor:
        img = self.left_eye_roi_image_generated()
        res = img * self._std.expand_as(img)
        res = res + self._mean.expand_as(res)
        return res

    @graph_node
    @torch.no_grad()
    def right_eye_roi_image_generated_destandardized(self) -> Tensor:
        img = self.right_eye_roi_image_generated()
        res = img * self._std.expand_as(img)
        res = res + self._mean.expand_as(res)
        return res

    @graph_node
    def mouth_roi_bboxes(self) -> Tensor:
        mouth_roi_bboxes = []
        loc_mouth = self.batch["mouth_bbox"]
        for b in range(loc_mouth.size(0)):
            mouth_roi_bboxes.append(
                torch.cat(
                    [
                        loc_mouth.new_full((1, 1), b),
                        loc_mouth[b : b + 1, :],
                    ],
                    dim=-1,
                )
            )
        mouth_roi_bboxes = torch.cat(mouth_roi_bboxes, dim=0).to(loc_mouth)
        return mouth_roi_bboxes

    @graph_node
    def mouth_roi_image_dst_normalized(self) -> Tensor:
        return roi_align(
            input=self.image_dst_normalized(),
            boxes=self.mouth_roi_bboxes(),
            output_size=self.mouth_roi_size,
        )

    @graph_node
    def mouth_roi_image_generated(self) -> Tensor:
        return roi_align(
            input=self.face_swapper_generate_image(),
            boxes=self.mouth_roi_bboxes(),
            output_size=self.mouth_roi_size,
        )


class GanPix2PixPipeline(BasePix2PixPipeline):
    def __init__(
        self,
        config: "GanPix2PixConfig",
        generator: "nn.Module",
        discriminator: "nn.Module",
        device: Union[str, torch.device],
        discriminator_left_eye: Optional["FacialComponentDiscriminator"] = None,
        discriminator_right_eye: Optional["FacialComponentDiscriminator"] = None,
        discriminator_mouth: Optional["FacialComponentDiscriminator"] = None,
    ):
        super().__init__(
            config=config,
            generator_model=generator,
            device=device,
        )
        self.config: "GanPix2PixConfig"
        self.discriminator = discriminator
        self.discriminator_left_eye = discriminator_left_eye
        self.discriminator_right_eye = discriminator_right_eye
        self.discriminator_mouth = discriminator_mouth

    @graph_node
    def discriminator_left_eye_fake(self) -> Tuple[Tensor, List[Tensor]]:
        out, out_feats = self.discriminator_left_eye(
            self.left_eye_roi_image_generated(), return_feats=True
        )
        return out, out_feats

    @graph_node
    def discriminator_right_eye_fake(self) -> Tuple[Tensor, List[Tensor]]:
        out, out_feats = self.discriminator_right_eye(
            self.right_eye_roi_image_generated(), return_feats=True
        )
        return out, out_feats

    @graph_node
    def discriminator_mouth_fake(self) -> Tuple[Tensor, List[Tensor]]:
        out, out_feats = self.discriminator_mouth(
            self.mouth_roi_image_generated(), return_feats=True
        )
        return out, out_feats

    @graph_node
    def discriminator_left_eye_dst(self) -> Tuple[Tensor, List[Tensor]]:
        out, out_feats = self.discriminator_left_eye(
            self.left_eye_roi_image_dst_normalized(), return_feats=True
        )
        return out, out_feats

    @graph_node
    def discriminator_right_eye_dst(self) -> Tuple[Tensor, List[Tensor]]:
        out, out_feats = self.discriminator_right_eye(
            self.right_eye_roi_image_dst_normalized(), return_feats=True
        )
        return out, out_feats

    @graph_node
    def discriminator_mouth_dst(self) -> Tuple[Tensor, List[Tensor]]:
        out, out_feats = self.discriminator_mouth(
            self.mouth_roi_image_dst_normalized(), return_feats=True
        )
        return out, out_feats

    @graph_node
    def discriminate_left_eye_fake(self) -> Tensor:
        return self.discriminator_left_eye(
            self.left_eye_roi_image_generated().detach()
        )[0]

    @graph_node
    def discriminate_right_eye_fake(self) -> Tensor:
        return self.discriminator_right_eye(
            self.right_eye_roi_image_generated().detach()
        )[0]

    @graph_node
    def discriminate_mouth_fake(self) -> Tensor:
        return self.discriminator_mouth(self.mouth_roi_image_generated().detach())[0]

    @graph_node
    def discriminate_left_eye_dst(self) -> Tensor:
        return self.discriminator_left_eye(self.left_eye_roi_image_dst_normalized())[0]

    @graph_node
    def discriminate_right_eye_dst(self) -> Tensor:
        return self.discriminator_right_eye(self.right_eye_roi_image_dst_normalized())[
            0
        ]

    @graph_node
    def discriminate_mouth_dst(self) -> Tensor:
        return self.discriminator_mouth(self.mouth_roi_image_dst_normalized())[0]


def pipeline_from_config(
    config: "BasePix2PixConfig",
    device: Union[str, torch.device],
) -> BasePix2PixPipeline:
    return BasePix2PixPipeline(
        config=config,
        generator_model=to_device(config.avaturn_swapper_model(), device=device),
        device=device,
    )
