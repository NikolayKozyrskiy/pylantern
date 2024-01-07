from contextlib import contextmanager
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
from kornia.color import rgb_to_lab
from matches.shortcuts.dag import graph_node
from torch import Tensor, nn

from pylantern import BasePipeline
from pylantern.common.constants import (
    DEFAULT_IMG_MEAN,
    DEFAULT_IMG_STD,
    IMG_8BIT_MAX_VAL,
)
from pylantern.common.nn.functional import rgb2gray
from pylantern.common.utils import to_device
from pylantern.common.utils.img import (
    crop_img_tensor_by_params,
    crop_random_params,
    interpolate_img_tensor,
)
from pylantern.common.utils.tensor import denormalize_tensor, normalize_tensor

if TYPE_CHECKING:
    from pylantern.tasks.gan.pix2pix.configs import BasePix2PixConfig


class BasePix2PixPipeline(BasePipeline):
    def __init__(
        self,
        config: "BasePix2PixConfig",
        generator_model: "nn.Module",
        discriminator_model: "nn.Module",
        device: Union[str, torch.device],
    ):
        BasePipeline.__init__(self, config=config, device=device)
        self.config: "BasePix2PixConfig"
        self.generator_model = generator_model
        self.discriminator_model = discriminator_model
        self._mean = (
            self.generator_model.mean
            if hasattr(self.generator_model, "mean")
            else torch.tensor(DEFAULT_IMG_MEAN).to(device)
        )
        self._std = (
            self.generator_model.std
            if hasattr(self.generator_model, "std")
            else torch.tensor(DEFAULT_IMG_STD).to(device)
        )
        self._max_orig_val = torch.tensor(IMG_8BIT_MAX_VAL, dtype=torch.float32).to(
            device
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
    @torch.no_grad()
    def generator_model_predict(self) -> Tensor:
        pred = self.generator_model.predict_image(self.image_src())
        return pred

    @graph_node
    def generator_model_forward(self) -> Tensor:
        pred = self.generator_model(self.image_src())
        return pred

    @graph_node
    def segmentation_mask_predict(self) -> Tensor:
        return self.generator_model_forward()[:, 3:, ...]

    @graph_node
    def segmentation_mask_predict_sigmoid(self) -> Tensor:
        return torch.sigmoid(self.segmentation_mask_predict())

    @graph_node
    def generator_model_forward_masked(self) -> Tensor:
        img = self.generator_model_forward()[:, :3, ...]
        mask = self.segmentation_mask_predict_sigmoid()
        img = img * mask + (1 - mask) * self.image_src_normalized()
        return img

    @graph_node
    def get_predicted_image(self) -> Tensor:
        pred = (
            self.generator_model_forward_masked()
            if self.config.predict_mask
            else self.generator_model_forward()
        )
        return pred

    @graph_node
    def get_predicted_image_destandardized(self) -> Tensor:
        img = self.get_predicted_image()
        res = img * self._std.expand_as(img)
        res = res + self._mean.expand_as(res)
        return torch.clip(res, min=0.0, max=1.0)

    @graph_node
    def get_predicted_image_lab(self) -> Tensor:
        img = self.get_predicted_image_destandardized()
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
            img=self.get_predicted_image(),
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
    def get_predicted_image_normalized_gray_128(self) -> Tensor:
        out_gray = rgb2gray(self.get_predicted_image(), keepdim=True)
        return interpolate_img_tensor(img=out_gray, size=(128, 128))

    @graph_node
    def image_dst_normalized_112(self) -> Tensor:
        return interpolate_img_tensor(img=self.image_dst_normalized(), size=(112, 112))

    @graph_node
    def get_predicted_image_normalized_112(self) -> Tensor:
        return interpolate_img_tensor(img=self.get_predicted_image(), size=(112, 112))

    @graph_node
    def image_dst_resized_scaled_224(self) -> Tensor:
        return interpolate_img_tensor(img=self.image_dst_scaled(), size=(224, 224))

    @graph_node
    def get_predicted_image_resized_destandardized_224(self) -> Tensor:
        return interpolate_img_tensor(
            img=self.get_predicted_image_destandardized(), size=(224, 224)
        )


def base_pix2pix_pipeline_from_config(
    config: "BasePix2PixConfig",
    device: Union[str, torch.device],
) -> BasePix2PixPipeline:
    return BasePix2PixPipeline(
        config=config,
        generator_model=to_device(config.generator_model(), device=device),
        discriminator_model=to_device(config.discriminator_model(), device=device),
        device=device,
    )
