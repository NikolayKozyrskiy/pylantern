from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from .image_mask_folders import ImageMaskFoldersDataset, change_dict_key


class FFHQDataset(ImageMaskFoldersDataset):
    def __init__(
        self,
        data_root: Path,
        fold_ids: Sequence[int | str],
        attributes: Mapping[str, str],
        split: str = "default",
        transforms: Callable = ToTensorV2(),
        eye_bbox_enlarge_ratio: float = 1.5,
    ) -> None:
        super().__init__(
            data_root=data_root,
            fold_ids=fold_ids,
            attributes=attributes,
            split=split,
            transforms=transforms,
        )
        self.eye_bbox_enlarge_ratio = eye_bbox_enlarge_ratio
        self._attr_loaders.update(
            {
                "kps_5_buffalo_l_src": self.load_fp64_npy,
                "kps_5_buffalo_l_dst": self.load_fp64_npy,
                "left_eye_bbox": self.load_left_eye_bbox,
                "right_eye_bbox": self.load_right_eye_bbox,
                "mouth_bbox": self.load_mouth_bbox,
            }
        )
        self._face_components = self._prepare_face_components_bboxes()

    def load_left_eye_bbox(self, attr: str, sample_name: str) -> np.ndarray:
        return self._load_face_component_bbox(sample_name=sample_name, part="left_eye")

    def load_right_eye_bbox(self, attr: str, sample_name: str) -> np.ndarray:
        return self._load_face_component_bbox(sample_name=sample_name, part="right_eye")

    def load_mouth_bbox(self, attr: str, sample_name: str) -> np.ndarray:
        return self._load_face_component_bbox(sample_name=sample_name, part="mouth")

    def _load_face_component_bbox(self, sample_name: str, part: str) -> np.ndarray:
        face_component = self._face_components[f"{int(sample_name):08d}"][part]
        mean = face_component[0:2]
        half_len = face_component[2]
        half_len = half_len * self.eye_bbox_enlarge_ratio
        loc = np.hstack((mean - half_len + 1, mean + half_len)).tolist()
        if part == "left_eye":
            loc.append("bboxes")
        else:
            loc.append(f"{part}_bbox")
        loc = [loc]
        return loc

    def _apply_transforms(self, item: dict, **transform_kwargs) -> dict:
        item = self._image_to_albu(item)
        item = self._bboxes_to_albu(item)
        item = self._kps_5_buffalo_l_to_albu(item)
        item: dict = self.transforms(**item, **transform_kwargs)
        item = self._image_from_albu(item)
        item = self._bboxes_from_albu(item)
        item = self._kps_5_buffalo_l_from_albu(item)
        return item

    def _bboxes_to_albu(self, item: dict) -> dict:
        if "left_eye_bbox" in item:
            item = change_dict_key(d=item, old_key="left_eye_bbox", new_key="bboxes")
        else:
            item["bboxes"] = []
        for k in "right_eye_bbox", "mouth_bbox":
            if k not in item.keys():
                item[k] = []
        return item

    def _bboxes_from_albu(self, item: dict) -> dict:
        if len(item["bboxes"]) > 0:
            item["left_eye_bbox"] = torch.tensor(
                item.pop("bboxes")[0][:4], dtype=torch.float32
            )
        else:
            del item["bboxes"]
        for k in "right_eye_bbox", "mouth_bbox":
            if len(item[k]) == 0:
                del item[k]
            else:
                item[k] = torch.tensor(item[k][0][:4], dtype=torch.float32)
        return item

    def _kps_5_buffalo_l_to_albu(self, item: dict) -> dict:
        if "kps_5_buffalo_l_dst" in item:
            item = change_dict_key(
                d=item, old_key="kps_5_buffalo_l_dst", new_key="keypoints"
            )
        else:
            item["keypoints"] = []
        for k in ["kps_5_buffalo_l_src"]:
            if k not in item.keys():
                item[k] = []
        return item

    def _kps_5_buffalo_l_from_albu(self, item: dict) -> dict:
        if len(item["keypoints"]) > 0:
            item["kps_5_buffalo_l_dst"] = torch.tensor(
                item.pop("keypoints"), dtype=torch.float64
            )
        else:
            del item["keypoints"]
        for k in ["kps_5_buffalo_l_src"]:
            if len(item[k]) == 0:
                del item[k]
            else:
                item[k] = torch.tensor(item[k], dtype=torch.float64)
        return item

    def _prepare_face_components_bboxes(self) -> Optional[dict]:
        face_components_bboxes = None
        if "left_eye_bbox" in self.attrs.keys():
            face_components_bboxes = torch.load(
                self.data_root / self.attrs["left_eye_bbox"], map_location="cpu"
            )
        if "right_eye_bbox" in self.attrs.keys() and face_components_bboxes is None:
            face_components_bboxes = torch.load(
                self.data_root / self.attrs["right_eye_bbox"], map_location="cpu"
            )
        if "mouth_bbox" in self.attrs.keys() and face_components_bboxes is None:
            face_components_bboxes = torch.load(
                self.data_root / self.attrs["mouth_bbox"], map_location="cpu"
            )
        return face_components_bboxes
