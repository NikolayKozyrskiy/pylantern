import warnings
from typing import Callable, List, Optional, Sequence, Tuple, Union

import albumentations as albu
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2

from pylantern.common.transforms.crop import CropByMask
from pylantern.common.transforms.degradation import DownscaleDegradation
from pylantern.common.transforms.flip import HorizontalFlipOrderAware
from pylantern.common.transforms.mask import FilterMask, SubtractMasks
from pylantern.common.transforms.pad import PadToRatio
from pylantern.common.transforms.resize import ResizeRandomAspectRatio
from pylantern.common.transforms.target_ignore import SpecificTargetIgnore
from pylantern.common.transforms.tensor import ToFloatTensor

warnings.filterwarnings(action="ignore", message=".*CoarseDropout.*")


def append_default_transforms(transforms: List[albu.BasicTransform]):
    return transforms + to_float_tensor()


def wrap_transforms(
    transforms: List[List[albu.BasicTransform]],
    transforms_preproc_fn: Callable = append_default_transforms,
    p: float = 0.5,
):
    first = compose(transforms_preproc_fn(transforms[0]))
    if len(transforms) == 1:
        return first
    elif len(transforms) == 2:
        second = compose(transforms_preproc_fn(transforms[1]))
        return albu.OneOrOther(first, second, p=p)
    else:
        raise NotImplementedError(
            f"OneOf composition for a list with transforms of size > 2 is not implemented yet"
        )


def compose(transforms: List[albu.BasicTransform]) -> albu.Compose:
    return albu.Compose(
        transforms=transforms,
        bbox_params=albu.BboxParams(format="pascal_voc"),
        keypoint_params=albu.KeypointParams(format="xy", remove_invisible=False),
        additional_targets={
            "image_dst": "image",
            "mask_src": "mask",
            "mask_blended_dst": "image",
            "kps_5_buffalo_l_src": "keypoints",
            "right_eye_bbox": "bboxes",  # "bboxes" is "left_eye_bbox"
            "mouth_bbox": "bboxes",
        },
    )


def to_tensor():
    return [ToTensorV2(transpose_mask=True)]


def to_float_tensor():
    return [ToFloatTensor(transpose_mask=True)]


def resize(
    dst_img_shape: Tuple[int, int],
    interpolation: int = cv2.INTER_AREA,
):
    return [
        albu.Resize(
            height=dst_img_shape[0],
            width=dst_img_shape[1],
            interpolation=interpolation,
            always_apply=True,
        )
    ]


def train_pixel_heavy_augs(
    degradation_scale_min: float = 0.25,
    degradation_scale_max: float = 0.99,
    blur_limit: int = 7,
    compression_quality: int = 70,
):
    return [
        SpecificTargetIgnore(
            downscale_degradation(
                scale_min=degradation_scale_min,
                scale_max=degradation_scale_max,
                p=0.9,
            )
            + blurs(blur_limit=blur_limit, allow_shifted=True)
            + [
                albu.ColorJitter(),
                albu.ISONoise(),
                albu.ImageCompression(quality_lower=compression_quality),
            ],
            specific_targets=["mask_blended_dst"],
        ),
    ]


def train_pixel_lite_augs(
    degradation_scale_min: float = 0.25,
    degradation_scale_max: float = 0.90,
    blur_limit: int = 3,
):
    return [
        SpecificTargetIgnore(
            [
                SpecificTargetIgnore(
                    downscale_degradation(
                        scale_min=degradation_scale_min,
                        scale_max=degradation_scale_max,
                        p=0.9,
                    )
                    + blurs(blur_limit=blur_limit, allow_shifted=True),
                    specific_targets=["image_dst"],
                ),
                albu.ColorJitter(
                    brightness=0.25,
                    contrast=0.25,
                    saturation=0.25,
                    hue=0.1 / np.pi,
                )
                # albu.ImageCompression(quality_lower=95),
            ],
            specific_targets=["mask_blended_dst"],
        ),
    ]


def train_spatial_augs_aligned(
    rotate_limit: int = 3,
    scale_limit: float = 0.01,
    shift_limit: float = 0.01,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    apply_optical_distortion: bool = True,
):
    augs = [
        HorizontalFlipOrderAware(
            swap_pairs=[["bboxes", "right_eye_bbox"]],
            p=0.5,
        )
    ]
    if apply_optical_distortion:
        augs += [
            albu.OpticalDistortion(
                distort_limit=0.01,
                shift_limit=0,
                interpolation=interpolation,
                border_mode=border_mode,
                value=0.0,
                mask_value=0,
                p=0.5,
            )
        ]
    augs += [
        albu.ShiftScaleRotate(
            shift_limit=shift_limit,
            scale_limit=scale_limit,
            rotate_limit=rotate_limit,
            interpolation=interpolation,
            border_mode=border_mode,
            value=0.0,
            mask_value=0.0,
            p=0.5,
        )
    ]
    return augs


def train_spatial_augs(
    rotate_limit: int = 60,
    scale_limit: float = 0.1,
    shift_limit: float = 0.0625,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
):
    return [
        HorizontalFlipOrderAware(
            swap_pairs=[["bboxes", "right_eye_bbox"]],
            p=0.5,
        ),
        albu.OneOf(
            [
                albu.OpticalDistortion(
                    interpolation=interpolation,
                    border_mode=border_mode,
                    value=0.0,
                    mask_value=0,
                ),
                albu.GridDistortion(
                    interpolation=interpolation,
                    border_mode=border_mode,
                    value=0.0,
                    mask_value=0,
                ),
                albu.ElasticTransform(
                    interpolation=interpolation,
                    border_mode=border_mode,
                    value=0.0,
                    mask_value=0,
                ),
            ],
            p=0.7,
        ),
        albu.ShiftScaleRotate(
            shift_limit=shift_limit,
            scale_limit=scale_limit,
            rotate_limit=rotate_limit,
            interpolation=interpolation,
            border_mode=border_mode,
            value=0.0,
            mask_value=0.0,
            p=0.8,
        ),
    ]


def downscale_degradation(
    scale_min: float = 0.5,
    scale_max: float = 0.99,
    always_apply: bool = False,
    p: float = 0.5,
) -> List[DownscaleDegradation]:
    return [
        DownscaleDegradation(
            scale_min=scale_min,
            scale_max=scale_max,
            always_apply=always_apply,
            p=p,
        )
    ]


def blurs(
    blur_limit: int = 7,
    allow_shifted: bool = True,
    p: float = 0.5,
) -> List[albu.OneOf]:
    return [
        albu.OneOf(
            transforms=[
                albu.Blur(blur_limit=blur_limit),
                albu.MotionBlur(blur_limit=blur_limit, allow_shifted=allow_shifted),
            ],
            p=p,
        )
    ]


def full_image_augs(image_size):
    return [
        albu.LongestMaxSize(426, p=1),
        albu.PadIfNeeded(
            *image_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
        ),
    ]


def scale_xy_transform(range):
    return [ResizeRandomAspectRatio(range)]


def optical_distortion_transform(
    distort_limit=0.1,
    shift_limit=0.05,
    padding_value=0.0,
    mask_value=0.0,
    p=0.5,
):
    return [
        albu.OpticalDistortion(
            distort_limit=distort_limit,
            shift_limit=shift_limit,
            border_mode=cv2.BORDER_CONSTANT,
            value=padding_value,
            mask_value=mask_value,
            p=p,
        )
    ]


def crop_by_mask_augs(image_hw, mask_name="mask"):
    h, w = image_hw
    return [
        albu.Sequential(
            [CropByMask(mask_name=mask_name), PadToRatio(h / w), albu.Resize(h, w)]
        )
    ]


def filter_mask(
    keep: Optional[Sequence[int]] = None,
    remove: Optional[Sequence[int]] = None,
    mask_name: str = "mask",
):
    return [FilterMask(keep, remove, mask_name)]


def subtract_masks(
    primary_mask: str = "mask", secondary_masks: Optional[Sequence[str]] = None
):
    return [SubtractMasks(primary_mask, secondary_masks)]
