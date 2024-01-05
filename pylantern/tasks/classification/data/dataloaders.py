from typing import TYPE_CHECKING

from pylantern.common.data.dataloaders import get_train_loader as get_train_loader_base
from pylantern.common.data.dataloaders import (
    get_validation_loader as get_validation_loader_base,
)

from .dataset import get_train_dataset, get_validation_dataset

if TYPE_CHECKING:
    from matches.loop.loader_scheduling import DataloaderSchedulerWrapper
    from torch.utils.data import DataLoader

    from ..config import ClassificationConfig


def get_train_loader(config: "ClassificationConfig") -> "DataloaderSchedulerWrapper":
    return get_train_loader_base(config=config, dataset=get_train_dataset(config))


def get_validation_loader(config: "ClassificationConfig") -> "DataLoader":
    return get_validation_loader_base(
        config=config, dataset=get_validation_dataset(config)
    )
