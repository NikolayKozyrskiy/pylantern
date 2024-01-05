import random
from typing import Dict, Tuple

import albumentations.augmentations.geometric.functional as F
import cv2
from albumentations import DualTransform


class ResizeRandomAspectRatio(DualTransform):
    def __init__(
        self, ratio_range: Tuple[int, int] = (1, 2), interpolation=cv2.INTER_LINEAR
    ):
        super().__init__()
        self.interpolation = interpolation
        self.ratio_range = ratio_range

    def get_params(self) -> Dict:
        ratio = random.uniform(*self.ratio_range)
        invert = random.choice((True, False))
        ratio = 1 / ratio if invert else ratio
        return {"ratio": ratio}

    def update_params(self, params, **kwargs):
        params = super().update_params(params, **kwargs)
        params.update(
            {
                "new_w": int(round(params["cols"] * params["ratio"])),
                "new_h": int(round(params["rows"] / params["ratio"])),
            }
        )
        return params

    def apply(self, img, interpolation=cv2.INTER_LINEAR, **params):
        return F.resize(
            img,
            height=params["new_h"],
            width=params["new_w"],
            interpolation=interpolation,
        )

    def apply_to_bbox(self, bbox, **params):
        # Bounding box coordinates are scale invariant
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        height = params["rows"]
        width = params["cols"]
        new_w = params["new_w"]
        new_h = params["new_h"]
        scale_x = new_w / width
        scale_y = new_h / height
        return F.keypoint_scale(keypoint, scale_x, scale_y)

    def get_transform_init_args_names(self):
        return "ratio_range", "interpolation"
