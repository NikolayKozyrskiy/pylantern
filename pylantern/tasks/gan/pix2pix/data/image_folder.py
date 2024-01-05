from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Union

import h5py
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from pylantern.common.utils.data import read_split
from pylantern.common.utils.img import load_img


def change_dict_key(d: dict, old_key: str, new_key: str) -> dict:
    d[new_key] = d.pop(old_key)
    return d


@lru_cache()
def open_h5(path: Path):
    return h5py.File(path, "r")


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        fold_ids: Sequence[int | str],
        attributes: Mapping[str, str],
        split: str = "default",
        transforms: Callable = ToTensorV2(),
        eye_bbox_enlarge_ratio: float = 1.5,
    ) -> None:
        super().__init__()
        self.eye_bbox_enlarge_ratio = eye_bbox_enlarge_ratio
        self.data_root = data_root
        self.attrs = attributes
        self.transforms = transforms
        # assert "image" in self.attrs, "image must be defined in attributes"

        self.sample_ids = read_split(Path(data_root) / f"splits/{split}", fold_ids)
        if isinstance(self.attrs, Sequence):
            self.attrs = {a: a for a in self.attrs}

        self._attr_loaders = {
            "image_src": self.load_image,  # input image
            "image_dst": self.load_image,  # gt image
            "mask_src": self.load_mask,  # input mask
            "mask": self.load_mask,  # gt binary mask
            "mask_blended_dst": self.load_fp32_npy,  # gt soft mask. self.load_soft_mask_h5
            "kps_5_buffalo_l_src": self.load_fp64_npy,
            "kps_5_buffalo_l_dst": self.load_fp64_npy,
            "left_eye_bbox": self.load_left_eye_bbox,
            "right_eye_bbox": self.load_right_eye_bbox,
            "mouth_bbox": self.load_mouth_bbox,
        }

        self._face_components = self._prepare_face_components_bboxes()

    def load_attrs(self, sample_name: str):
        return {
            attr_name: self._attr_loaders[attr_name](attr_dir, sample_name)
            for attr_name, attr_dir in self.attrs.items()
        }

    def load_soft_mask_h5(self, attr: str, sample_name: str) -> np.ndarray:
        attr, ext = attr.split(".")
        h5_file = open_h5(self.data_root / f"{attr}.h5")
        mask: np.ndarray = np.array(
            h5_file[f"mask_blended_dst/{sample_name}"], dtype=np.float32
        )
        return mask

    def load_image(self, attr: str, sample_name: str) -> np.ndarray:
        attr, ext = attr.split(".")
        img = load_img(
            self.data_root / f"{attr}/{sample_name}.{ext}", convert_bgr2rgb=True
        )
        return img.astype(np.uint8)

    def load_mask(self, attr: str, sample_name: str) -> np.ndarray:
        attr, ext = attr.split(".")
        mask = load_img(self.data_root / f"{attr}/{sample_name}.{ext}")
        if mask.ndim == 3 and mask.shape[-1] == 4:
            mask = mask[..., 3]
        if mask.ndim == 2:
            mask = mask[..., None]
        return mask.astype(np.uint8)

    def load_fp32_npy(self, attr: str, sample_name: str) -> np.ndarray:
        return self._load_npy(attr=attr, sample_name=sample_name).astype(np.float32)

    def load_fp64_npy(self, attr: str, sample_name: str) -> np.ndarray:
        return self._load_npy(attr=attr, sample_name=sample_name).astype(np.float64)

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

    def _load_npy(self, attr: str, sample_name: str) -> np.ndarray:
        parts = attr.split(".")
        attr = parts[0]
        ext = parts[1] if len(parts) == 2 else "npy"
        return np.load(self.data_root / f"{attr}/{sample_name}.{ext}")

    def get(self, sample_name: str, **transform_kwargs):
        item = {
            "name": sample_name,
            **self.load_attrs(sample_name),
        }
        item = self._apply_transforms(item, **transform_kwargs)
        return item

    def __getitem__(self, index: Union[str, int]) -> dict:
        if isinstance(index, int):
            index = self.sample_ids[index]
        return self.get(index)

    def __len__(self):
        return len(self.sample_ids)

    def _apply_transforms(self, item: dict, **transform_kwargs) -> dict:
        item = self._image_to_albu(item)
        item = self._bboxes_to_albu(item)
        item = self._kps_5_buffalo_l_to_albu(item)
        item: dict = self.transforms(**item, **transform_kwargs)
        item = self._image_from_albu(item)
        item = self._bboxes_from_albu(item)
        item = self._kps_5_buffalo_l_from_albu(item)
        return item

    def _image_to_albu(self, item: dict) -> dict:
        return change_dict_key(d=item, old_key="image_src", new_key="image")

    def _image_from_albu(self, item: dict) -> dict:
        return change_dict_key(d=item, old_key="image", new_key="image_src")

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
