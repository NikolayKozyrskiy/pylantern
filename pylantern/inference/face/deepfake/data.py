from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Optional, Tuple

import numpy as np
from insightface.app.common import Face


class CropPasteMethod(str, Enum):
    INFA_INSWAPPER = "infa_inswapper"
    INFA_GENERATOR = "infa_generator"
    FACEXLIB = "facexlib"
    BBOX = "bbox"
    DEFAULT = "infa_inswapper"


class IOTypes(str, Enum):
    INPUT = "Input"
    SWAPPED = "Swapped"
    SWAPPED_ENHANCED = "Swapped_Enhanced"
    ENHANCED = "Enhanced"


@dataclass
class FaceInfaProcessed:
    img: np.ndarray
    face: Optional[Face]

    @cached_property
    def bbox_orig(self) -> Optional[Tuple[int, int, int, int]]:
        if self.face is not None:
            return tuple(int(round(self.face["bbox"][i])) for i in range(4))
        return None

    @cached_property
    def crop_orig_size(self) -> Optional[Tuple[int, int]]:
        if self.bbox_orig is not None:
            w = self.bbox_orig[2] - self.bbox_orig[0]
            h = self.bbox_orig[3] - self.bbox_orig[1]
            return w, h
        return None

    @cached_property
    def bbox_squared(self) -> Optional[Tuple[int, int, int, int]]:
        if self.bbox_orig is not None:
            x_l, y_l, x_r, y_r = self.bbox_orig
            x_abs = abs(x_r - x_l)
            y_abs = abs(y_r - y_l)
            if x_abs < y_abs:
                diff = y_abs - x_abs
                x_l -= diff / 2
                x_r += diff / 2
                x_l = int(x_l)
                x_r = int(x_r)
            else:
                diff = x_abs - y_abs
                y_l -= diff / 2
                y_r += diff / 2
                y_l = int(y_l)
                y_r = int(y_r)
            return x_l, y_l, x_r, y_r
        return None

    @cached_property
    def crop_squared_size(self) -> Optional[Tuple[int, int]]:
        if self.bbox_squared is not None:
            w = self.bbox_squared[2] - self.bbox_squared[0]
            h = self.bbox_squared[3] - self.bbox_squared[1]
            return w, h
        return None


@dataclass
class FaceSwapData:
    src_img: Optional[np.ndarray] = None
    dst_img: Optional[np.ndarray] = None
    src_face: Optional[Face] = None
    src_face_idx: Optional[int] = None
    dst_face: Optional[Face] = None
    dst_face_idx: Optional[int] = None
    dst_face_processed: Optional[FaceInfaProcessed] = None
    aligned_src_img: Optional[np.ndarray] = None
    aligned_dst_img: Optional[np.ndarray] = None
    transform_matrix: Optional[np.ndarray] = None
    predicted_dst_img: Optional[np.ndarray] = None
    predicted_dst_mask: Optional[np.ndarray] = None
    swapped_dst_img: Optional[np.ndarray] = None
    swapping_occurred: bool = False
    enhanced_swapped_dst_img: Optional[np.ndarray] = None
    enhanced_dst_img: Optional[np.ndarray] = None
