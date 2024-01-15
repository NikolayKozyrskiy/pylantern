from typing import TYPE_CHECKING, List, Optional, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.utils import face_align

from pylantern.common.utils.img import choose_interpolation, crop_by_bbox
from pylantern.inference.face.deepfake.data import FaceInfaProcessed, FaceSwapData

if TYPE_CHECKING:
    from pylantern.inference.face.deepfake.df_config import DeepFakeInferenceConfig


class FaceAlignerInfa:
    def __init__(
        self, input_face_size: Tuple[int, int], face_analyzer: "FaceAnalysis"
    ) -> None:
        self.face_analyzer = face_analyzer
        self._input_face_sizes = input_face_size

    def get_face_processed(
        self, img: np.ndarray, face_idx: Optional[int] = None
    ) -> "FaceInfaProcessed":
        face = self.get_face(img=img, face_idx=face_idx)
        return FaceInfaProcessed(img=img, face=face)

    def get_face(
        self, img: np.ndarray, face_idx: Optional[int] = None
    ) -> Optional["Face"]:
        if face_idx is None or face_idx == 0:
            return self.get_biggest_face(img=img)
        else:
            return self.get_face_by_idx(img=img, face_idx=face_idx)

    def get_faces(self, img: np.ndarray) -> Optional[List["Face"]]:
        """
        get faces from left to right by order
        """
        try:
            faces = self.face_analyzer.get(img)
            return sorted(faces, key=lambda x: x.bbox[0])
        except IndexError:
            return None

    def get_biggest_face(self, img: np.ndarray) -> Optional["Face"]:
        """
        get the biggest area face
        """
        face = self.face_analyzer.get(img, max_num=1)
        try:
            return face[0]
        except IndexError:
            return None

    def get_face_by_idx(self, img: np.ndarray, face_idx: int) -> Optional["Face"]:
        """
        get one face out of faces sorted from left to right
        """
        try:
            faces = self.face_analyzer.get(img)
            return sorted(faces, key=lambda x: x.bbox[0])[face_idx]
        except IndexError:
            return None

    def get_one_aligned_img(
        self,
        img: np.ndarray,
        face: Optional["Face"] = None,
        face_idx: Optional[int] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        face = self.get_face(img=img, face_idx=face_idx) if face is None else face
        if face is None:
            return None
        aligned_img, transform_matrix = face_align.norm_crop2(
            img, face.kps, self._input_face_sizes[0]
        )
        return aligned_img, transform_matrix

    def get_arcface_kps_biggest_img(self, img: np.ndarray) -> Optional[np.ndarray]:
        bboxes, kpss = self.face_analyzer.det_model.detect(
            img, max_num=1, metric="default"
        )
        return kpss[0] if bboxes.shape[0] != 0 else None

    def get_crop_by_bbox_orig(
        self,
        face_processed: "FaceInfaProcessed",
        crop_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[np.ndarray]:
        return crop_by_bbox(
            img=face_processed.img,
            bbox=face_processed.bbox_orig,
            crop_size=crop_size,
        )

    def get_crop_by_bbox_squared(
        self,
        face_processed: "FaceInfaProcessed",
        crop_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[np.ndarray]:
        return crop_by_bbox(
            img=face_processed.img,
            bbox=face_processed.bbox_squared,
            crop_size=crop_size,
        )

    def paste_back_by_bbox_squared(self, face_swap_data: "FaceSwapData") -> np.ndarray:
        if not face_swap_data.swapping_occurred:
            return face_swap_data.dst_img

        dsize_crop = face_swap_data.dst_face_processed.crop_squared_size
        predicted_img_resized = cv2.resize(
            face_swap_data.predicted_dst_img.astype(np.float32),
            dsize=dsize_crop,
            interpolation=choose_interpolation(
                img=face_swap_data.predicted_dst_img,
                dsize=dsize_crop,
            ),
        ).astype(np.float32)
        predicted_mask_resized = cv2.resize(
            face_swap_data.predicted_dst_mask.astype(np.float32),
            dsize=dsize_crop,
            interpolation=choose_interpolation(
                img=face_swap_data.predicted_dst_img,
                dsize=dsize_crop,
            ),
        )[..., None].astype(np.float32)
        orig_img_crop = self.get_crop_by_bbox_squared(
            face_processed=face_swap_data.dst_face_processed,
            crop_size=None,
        ).astype(np.float32)

        blended_result = (
            predicted_img_resized * predicted_mask_resized
            + (1.0 - predicted_mask_resized) * orig_img_crop
        )
        blended_result = blended_result.clip(0, 255).round().astype(np.uint8)

        swapped_dst_img = face_swap_data.dst_img.copy()
        x_l, y_l, x_r, y_r = face_swap_data.dst_face_processed.bbox_squared
        swapped_dst_img[y_l:y_r, x_l:x_r, :] = blended_result

        return swapped_dst_img

    def paste_back_infa(self, face_swap_data: "FaceSwapData") -> np.ndarray:
        fake_diff = face_swap_data.swapped_dst_img.astype(
            np.float32
        ) - face_swap_data.aligned_dst_img.astype(np.float32)
        fake_diff = np.abs(fake_diff).mean(axis=2)
        fake_diff[:2, :] = 0
        fake_diff[-2:, :] = 0
        fake_diff[:, :2] = 0
        fake_diff[:, -2:] = 0
        IM = cv2.invertAffineTransform(face_swap_data.transform_matrix)
        img_white = np.full(
            (
                face_swap_data.aligned_dst_img.shape[0],
                face_swap_data.aligned_dst_img.shape[1],
            ),
            255,
            dtype=np.float32,
        )
        swapped_img = cv2.warpAffine(
            face_swap_data.swapped_dst_img,
            IM,
            (face_swap_data.dst_img.shape[1], face_swap_data.dst_img.shape[0]),
            borderValue=0.0,
        )
        img_white = cv2.warpAffine(
            img_white,
            IM,
            (face_swap_data.dst_img.shape[1], face_swap_data.dst_img.shape[0]),
            borderValue=0.0,
        )
        fake_diff = cv2.warpAffine(
            fake_diff,
            IM,
            (face_swap_data.dst_img.shape[1], face_swap_data.dst_img.shape[0]),
            borderValue=0.0,
        )
        img_white[img_white > 20] = 255
        fthresh = 10
        fake_diff[fake_diff < fthresh] = 0
        fake_diff[fake_diff >= fthresh] = 255
        img_mask = img_white
        mask_h_inds, mask_w_inds = np.where(img_mask == 255)
        mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
        mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
        mask_size = int(np.sqrt(mask_h * mask_w))
        k = max(mask_size // 10, 10)
        kernel = np.ones((k, k), np.uint8)
        img_mask = cv2.erode(img_mask, kernel, iterations=1)
        kernel = np.ones((2, 2), np.uint8)
        fake_diff = cv2.dilate(fake_diff, kernel, iterations=1)
        k = max(mask_size // 20, 5)
        kernel_size = (k, k)
        blur_size = tuple(2 * i + 1 for i in kernel_size)
        img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
        k = 5
        kernel_size = (k, k)
        blur_size = tuple(2 * i + 1 for i in kernel_size)
        fake_diff = cv2.GaussianBlur(fake_diff, blur_size, 0)
        img_mask /= 255
        fake_diff /= 255
        img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])
        fake_merged = img_mask * swapped_img + (
            1 - img_mask
        ) * face_swap_data.dst_img.astype(np.float32)
        fake_merged = fake_merged.clip(0, 255).round().astype(np.uint8)
        return fake_merged

    def paste_back_aligned(self, face_swap_data: "FaceSwapData") -> np.ndarray:
        IM = cv2.invertAffineTransform(face_swap_data.transform_matrix)
        predicted_dst_img = cv2.warpAffine(
            face_swap_data.predicted_dst_img,
            IM,
            (face_swap_data.dst_img.shape[1], face_swap_data.dst_img.shape[0]),
            # borderValue=0.0,
            borderMode=cv2.BORDER_REPLICATE,
        )
        predicted_dst_mask = cv2.warpAffine(
            face_swap_data.predicted_dst_mask,
            IM,
            (face_swap_data.dst_img.shape[1], face_swap_data.dst_img.shape[0]),
            borderValue=0.0,
            borderMode=cv2.BORDER_CONSTANT,
        )[..., None]
        fake_merged = predicted_dst_img.astype(np.float32) * predicted_dst_mask + (
            1 - predicted_dst_mask
        ) * face_swap_data.dst_img.astype(np.float32)
        fake_merged = fake_merged.clip(0, 255).round().astype(np.uint8)
        return fake_merged


def load_face_aligner_infa(
    input_face_size: Tuple[int, int], face_analyzer: "FaceAnalysis"
) -> "FaceAlignerInfa":
    return FaceAlignerInfa(input_face_size=input_face_size, face_analyzer=face_analyzer)
