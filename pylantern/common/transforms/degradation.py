import random
import warnings
from typing import Any, Dict, Optional, Tuple

import albumentations as albu
import cv2
import numpy as np
from albumentations import functional as F

warnings.filterwarnings(action="ignore", message=".*CoarseDropout.*")


class DownscaleDegradation(albu.ImageOnlyTransform):
    """Decreases image quality by downscaling and upscaling back.
    Args:
        scale_min: minimum scale value, must be less than 1
        scale_max: maximum scale value, must be equal or less than 1 and bigger than scale_min

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        scale_min: float = 0.5,
        scale_max: float = 0.99,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        if scale_min > scale_max:
            raise ValueError(
                "Expected scale_min be less or equal scale_max, got {} {}".format(
                    scale_min, scale_max
                )
            )
        if scale_max > 1:
            raise ValueError(
                "Expected scale_max to be less than 1, got {}".format(scale_max)
            )
        self.scale_min = scale_min
        self.scale_max = scale_max
        self._interpolations = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
        ]

    def apply(
        self,
        img: np.ndarray,
        scale: Optional[float] = None,
        down_interpolation: Optional[int] = None,
        up_interpolation: Optional[int] = None,
        **params
    ) -> np.ndarray:
        return F.downscale(
            img,
            scale=scale,
            down_interpolation=down_interpolation,
            up_interpolation=up_interpolation,
        )

    def get_params(self) -> Dict[str, float]:
        return {
            "scale": random.uniform(self.scale_min, self.scale_max),
            "down_interpolation": self._get_random_interpolation(),
            "up_interpolation": self._get_random_interpolation(),
        }

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return "scale_min", "scale_max"

    def _get_random_interpolation(self) -> int:
        return np.random.choice(self._interpolations, size=1)[0]
