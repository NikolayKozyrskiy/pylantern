from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

from pylantern.common.data.dataloaders import get_eval_loader as get_eval_loader_base
from pylantern.common.data.dataloaders import get_train_loader as get_train_loader_base
from pylantern.common.data.dataloaders import (
    get_validation_loader as get_validation_loader_base,
)
from pylantern.tasks.gan.common.transforms.common import wrap_transforms

from .constants import DatasetType
from .datasets import FFHQDataset, ImageMaskFoldersDataset

if TYPE_CHECKING:
    from matches.loop.loader_scheduling import DataloaderSchedulerWrapper
    from torch.utils.data import DataLoader

    from ..configs import BasePix2PixConfig, FacePix2PixConfig

ConfigType = Union["BasePix2PixConfig", "FacePix2PixConfig"]


def get_train_loader(config: ConfigType) -> "DataloaderSchedulerWrapper":
    if config.dataset_type == DatasetType.IMAGE_MASK_FOLDERS:
        dataset = ImageMaskFoldersDataset(
            data_root=config.root_path,
            attributes=config.attrs_to_load,
            fold_ids=config.train_folds,
            split=config.split,
            transforms=wrap_transforms(config.train_transforms)
            if isinstance(config.train_transforms, list)
            else config.train_transforms,
        )
    elif config.dataset_type == DatasetType.FFHQ:
        dataset = FFHQDataset(
            data_root=config.root_path,
            attributes=config.attrs_to_load,
            fold_ids=config.train_folds,
            split=config.split,
            transforms=wrap_transforms(config.train_transforms)
            if isinstance(config.train_transforms, list)
            else config.train_transforms,
            eye_bbox_enlarge_ratio=config.eye_bbox_enlarge_ratio,
        )
    else:
        raise NotImplementedError

    return get_train_loader_base(config=config, dataset=dataset)


def get_validation_loader(config: ConfigType) -> "DataLoader":
    if config.dataset_type == DatasetType.IMAGE_MASK_FOLDERS:
        dataset = ImageMaskFoldersDataset(
            data_root=config.root_path,
            attributes=config.attrs_to_load,
            fold_ids=config.valid_folds,
            split=config.split,
            transforms=wrap_transforms(config.valid_transforms)
            if isinstance(config.valid_transforms, list)
            else config.valid_transforms,
        )
    elif config.dataset_type == DatasetType.FFHQ:
        dataset = FFHQDataset(
            data_root=config.root_path,
            attributes=config.attrs_to_load,
            fold_ids=config.valid_folds,
            split=config.split,
            transforms=wrap_transforms(config.valid_transforms)
            if isinstance(config.valid_transforms, list)
            else config.valid_transforms,
            eye_bbox_enlarge_ratio=config.eye_bbox_enlarge_ratio,
        )
    else:
        raise NotImplementedError

    return get_validation_loader_base(config=config, dataset=dataset)


def get_eval_loader(
    config: ConfigType,
    data_root: Optional[Path] = None,
    fold_ids: Optional[List[str]] = None,
    split: Optional[str] = None,
) -> "DataLoader":
    if config.dataset_type == DatasetType.IMAGE_MASK_FOLDERS:
        dataset = ImageMaskFoldersDataset(
            data_root=data_root if data_root is not None else config.root_path,
            attributes=config.attrs_to_load,
            fold_ids=fold_ids if fold_ids is not None else config.valid_folds,
            split=split if split is not None else config.split,
            transforms=wrap_transforms(config.valid_transforms)
            if isinstance(config.valid_transforms, list)
            else config.valid_transforms,
        )
    elif config.dataset_type == DatasetType.FFHQ:
        dataset = FFHQDataset(
            data_root=data_root if data_root is not None else config.root_path,
            attributes=config.attrs_to_load,
            fold_ids=fold_ids if fold_ids is not None else config.valid_folds,
            split=split if split is not None else config.split,
            transforms=wrap_transforms(config.valid_transforms)
            if isinstance(config.valid_transforms, list)
            else config.valid_transforms,
            eye_bbox_enlarge_ratio=config.eye_bbox_enlarge_ratio,
        )
    else:
        raise NotImplementedError

    return get_eval_loader_base(config=config, dataset=dataset)
