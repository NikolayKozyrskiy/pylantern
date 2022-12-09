from typing import NamedTuple, Optional, Callable, List, Dict, Union, Tuple, Any
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL.Image import Image

from ..config import ClassificationDatasetName
from ..transforms import train_basic_augs


class ClassificationDatasetItem(NamedTuple):
    image: Union[torch.Tensor, Image, np.ndarray]
    label: Union[torch.Tensor, int]
    name: Optional[str]


class ClassificationDatasetW(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        dataset_name: ClassificationDatasetName,
        image_hw: Tuple[int, int],
        transforms: List[Callable],
        label_name_map: Dict[ClassificationDatasetName, Dict[int, str]],
    ):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.image_hw = image_hw
        self.transforms = transforms
        self.label_name_map = label_name_map
        if label_name_map.get(dataset_name, None) is not None:
            self._get_fn = self._get_label_named
        else:
            self._get_fn = self._get

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> ClassificationDatasetItem:
        return self._get_fn(index)

    def _get_label_named(self, index: int) -> ClassificationDatasetItem:
        img, target = self.dataset[index]
        return ClassificationDatasetItem(
            img,
            target,
            f"{index}_{self.label_name_map[self.dataset_name][target]}",
        )

    def _get(self, index: int) -> ClassificationDatasetItem:
        img, target = self.dataset[index]
        return ClassificationDatasetItem(
            img,
            target,
            f"{index}",
        )
