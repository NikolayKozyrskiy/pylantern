from typing import NamedTuple, Optional, Callable, List, Dict, Union, Tuple, Any
from pathlib import Path
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from PIL.Image import Image

from pylantern.common.transforms import wrap_transforms

from ..config import ClassificationDatasetName, ClassificationConfig
from .label_name_map import LABEL_NAME_MAP


class ClassificationDatasetItem(NamedTuple):
    image: Union[torch.Tensor, Image, np.ndarray]
    label: Union[torch.Tensor, int]
    name: Optional[str]


class ClassificationDatasetW(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        dataset_name: ClassificationDatasetName,
        image_hw: Optional[Tuple[int, int]] = None,
    ):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.image_hw = image_hw
        self.label_name_map: Optional[Dict[int, str]] = LABEL_NAME_MAP.get(
            dataset_name, None
        )
        self._get_fn = (
            self._get if self.label_name_map is None else self._get_label_named
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> ClassificationDatasetItem:
        return self._get_fn(index)

    def _get_label_named(self, index: int) -> ClassificationDatasetItem:
        img, target = self.dataset[index]
        return ClassificationDatasetItem(
            img,
            target,
            f"{index}_{self.label_name_map[target]}_label_gt_{target}",
        )

    def _get(self, index: int) -> ClassificationDatasetItem:
        img, target = self.dataset[index]
        return ClassificationDatasetItem(
            img,
            target,
            f"{index}_label_gt_{target}",
        )


def get_train_dataset(config: ClassificationConfig) -> ClassificationDatasetW:
    os.makedirs(config.data_root, exist_ok=True)
    if config.dataset_name == ClassificationDatasetName.CIFAR10:
        dataset = datasets.CIFAR10(
            root=config.data_root,
            train=True,
            download=True,
            transform=wrap_transforms(config.train_transforms),
        )
    elif config.dataset_name == ClassificationDatasetName.CIFAR100:
        dataset = datasets.CIFAR100(
            root=config.data_root,
            train=True,
            download=True,
            transform=wrap_transforms(config.train_transforms),
        )
    elif config.dataset_name == ClassificationDatasetName.IMAGENET:
        dataset = datasets.ImageNet(
            root=config.data_root,
            split="train",
            transform=wrap_transforms(config.train_transforms),
        )
    else:
        raise ValueError(f"Given dataset {config.dataset_name} is not supported!")
    return ClassificationDatasetW(
        dataset=dataset,
        dataset_name=config.dataset_name,
        image_hw=config.image_hw,
    )


def get_validation_dataset(config: ClassificationConfig) -> ClassificationDatasetW:
    os.makedirs(config.data_root, exist_ok=True)
    if config.dataset_name == ClassificationDatasetName.CIFAR10:
        dataset = datasets.CIFAR10(
            root=config.data_root,
            train=False,
            download=True,
            transform=wrap_transforms(config.valid_transforms),
        )
    elif config.dataset_name == ClassificationDatasetName.CIFAR100:
        dataset = datasets.CIFAR100(
            root=config.data_root,
            train=False,
            download=True,
            transform=wrap_transforms(config.valid_transforms),
        )
    elif config.dataset_name == ClassificationDatasetName.IMAGENET:
        dataset = datasets.ImageNet(
            root=config.data_root,
            split="val",
            transform=wrap_transforms(config.valid_transforms),
        )
    else:
        raise ValueError(f"Given dataset {config.dataset_name} is not supported!")
    return ClassificationDatasetW(
        dataset=dataset,
        dataset_name=config.dataset_name,
        image_hw=config.image_hw,
    )
