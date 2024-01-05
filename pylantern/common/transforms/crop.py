from __future__ import annotations

import random

import numpy as np
from albumentations import (
    DualTransform,
    bbox_crop,
    clamping_crop,
    crop,
    crop_keypoint_by_coords,
)


class CropByMask(DualTransform):
    def __init__(
        self,
        rel_margins_top_left_bottom_right=(0.0, 0.0, 0.0, 0.0),
        ignore_values=None,
        ignore_channels=None,
        mask_name="mask",
        always_apply=False,
        p=1.0,
    ):
        super().__init__(always_apply, p)
        assert np.all(
            (np.array(rel_margins_top_left_bottom_right) >= 0)
            & (np.array(rel_margins_top_left_bottom_right) <= 1)
        )
        self.mask_name = mask_name

        self.rel_margins_top_left_bottom_right = rel_margins_top_left_bottom_right
        if ignore_values is not None and not isinstance(ignore_values, list):
            raise ValueError(
                "Expected `ignore_values` of type `list`, got `{}`".format(
                    type(ignore_values)
                )
            )
        if ignore_channels is not None and not isinstance(ignore_channels, list):
            raise ValueError(
                "Expected `ignore_channels` of type `list`, got `{}`".format(
                    type(ignore_channels)
                )
            )

        self.ignore_values = ignore_values
        self.ignore_channels = ignore_channels

    def get_params_dependent_on_targets(self, params):
        return "x_min", "x_max", "y_min", "y_max"

    def apply(self, img, x_min=0, x_max=0, y_min=0, y_max=0, **params):
        return crop(img, x_min, y_min, x_max, y_max)

    def apply_to_bbox(self, bbox, x_min=0, x_max=0, y_min=0, y_max=0, **params):
        return bbox_crop(
            bbox,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            rows=params["rows"],
            cols=params["cols"],
        )

    def apply_to_keypoint(self, keypoint, x_min=0, x_max=0, y_min=0, y_max=0, **params):
        return crop_keypoint_by_coords(
            keypoint, crop_coords=(x_min, y_min, x_max, y_max)
        )

    def _preprocess_mask(self, mask):
        if self.ignore_values is not None:
            ignore_values_np = np.array(self.ignore_values)
            mask = np.where(np.isin(mask, ignore_values_np), 0, mask)

        if mask.ndim == 3 and self.ignore_channels is not None:
            target_channels = np.array(
                [ch for ch in range(mask.shape[-1]) if ch not in self.ignore_channels]
            )
            mask = np.take(mask, target_channels, axis=-1)

        return mask

    def update_params(self, params, **kwargs):
        if self.mask_name in kwargs:
            mask = self._preprocess_mask(kwargs[self.mask_name])
        elif "masks" in kwargs and len(kwargs["masks"]):
            masks = kwargs["masks"]
            mask = self._preprocess_mask(masks[0])
            for m in masks[1:]:
                mask |= self._preprocess_mask(m)
        else:
            raise RuntimeError("Can not find mask for CropNonEmptyMaskIfExists")

        mask_height, mask_width = mask.shape[:2]

        if mask.any():
            mask = mask.sum(axis=-1) if mask.ndim == 3 else mask
            i, j = np.nonzero(mask)
            y_min = i.min()
            x_min = j.min()
            y_max = i.max()
            x_max = j.max()

            box_h, box_w = y_max - y_min, x_max - x_min

            abs_margins_top_left_bottom_right = (
                (
                    np.array([box_h, box_w, box_h, box_w])
                    * self.rel_margins_top_left_bottom_right
                )
                .round()
                .astype("int")
                .clip(min=5)
            )
            # fmt: off
            y_min, x_min, y_max, x_max = \
                [y_min, x_min, y_max, x_max] \
                + abs_margins_top_left_bottom_right \
                * [-1, -1, 1, 1]
            # fmt: on

            x_min, x_max = np.clip([x_min, x_max], 0, mask_width)
            y_min, y_max = np.clip([y_min, y_max], 0, mask_height)
            params.update(
                {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}
            )
        else:
            print(f"Empty mask {kwargs['name']}")
            params.update(
                {"x_min": 0, "x_max": mask_width, "y_min": 0, "y_max": mask_height}
            )

        return params

    def get_transform_init_args_names(self):
        return (
            "ignore_values",
            "ignore_channels",
            "rel_margins_top_left_bottom_right",
            "mask_name",
        )


class RandomCropNearBBox(DualTransform):
    """Crop bbox from image with random shift by x,y coordinates

    Args:
        max_part_shift (float): float value in (0.0, 1.0) range. Default 0.3
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, max_part_shift=0.3, always_apply=False, p=1.0):
        super(RandomCropNearBBox, self).__init__(always_apply, p)
        self.max_part_shift = max_part_shift

    def apply(self, img, x_min=0, x_max=0, y_min=0, y_max=0, **params):
        return clamping_crop(img, x_min, y_min, x_max, y_max)

    def get_params_dependent_on_targets(self, params):
        h, w, _ = params["image"].shape
        left, top, right, bottom = params.get("crop_bbox", None) or [0, 0, w, h]

        bbox_h = bottom - top
        bbox_w = right - left

        mode = random.randrange(0, 16)
        if mode & 1 != 0:
            bottom += int(bbox_h * random.choice([0.25, 0.5, 0.75, 1]))
        if mode & 2 != 0:
            top -= int(bbox_h * random.choice([-0.3, -0.2, -0.1, 0.1, 0.2]))
        if mode & 4 != 0:
            right += int(bbox_w * random.choice([0.1, 0.2, 0.3, 0.4]))
        if mode & 8 != 0:
            left -= int(bbox_w * random.choice([0.1, 0.2, 0.3, 0.4]))

        left = max(0, left)
        right = min(w, right)
        top = max(0, top)
        bottom = min(h, bottom)

        return {"x_min": left, "x_max": right, "y_min": top, "y_max": bottom}

    def apply_to_bbox(self, bbox, x_min=0, x_max=0, y_min=0, y_max=0, **params):
        h_start = y_min
        w_start = x_min
        return bbox_crop(bbox, y_max - y_min, x_max - x_min, h_start, w_start, **params)

    def apply_to_keypoint(self, keypoint, x_min=0, x_max=0, y_min=0, y_max=0, **params):
        return crop_keypoint_by_coords(
            keypoint,
            crop_coords=(x_min, y_min, x_max, y_max),
        )

    @property
    def targets_as_params(self):
        return ["crop_bbox", "image"]

    def get_transform_init_args_names(self):
        return ("max_part_shift",)
