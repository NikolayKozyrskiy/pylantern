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


class ImageMaskFoldersDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        fold_ids: Sequence[int | str],
        attributes: Mapping[str, str],
        split: str = "default",
        transforms: Callable = ToTensorV2(),
    ) -> None:
        super().__init__()
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
        }

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

    def _apply_transforms(self, item: dict, **transform_kwargs) -> dict:
        item = self._image_to_albu(item)
        item: dict = self.transforms(**item, **transform_kwargs)
        item = self._image_from_albu(item)
        return item

    def _image_to_albu(self, item: dict) -> dict:
        return change_dict_key(d=item, old_key="image_src", new_key="image")

    def _image_from_albu(self, item: dict) -> dict:
        return change_dict_key(d=item, old_key="image", new_key="image_src")

    def __getitem__(self, index: Union[str, int]) -> dict:
        if isinstance(index, int):
            index = self.sample_ids[index]
        return self.get(index)

    def __len__(self):
        return len(self.sample_ids)
