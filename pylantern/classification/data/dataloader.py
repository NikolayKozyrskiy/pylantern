from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import ignite.distributed as idist
from ignite.distributed.auto import auto_dataloader
from matches.loop.loader_scheduling import DataloaderSchedulerWrapper

from ..config import ClassificationConfig
from .dataset import get_train_dataset, get_validation_dataset


def get_train_loader(config: ClassificationConfig) -> DataloaderSchedulerWrapper:
    loader = DataloaderSchedulerWrapper(
        auto_dataloader(
            get_train_dataset(config),
            num_workers=config.train_loader_workers,
            batch_size=config.batch_size_train * idist.get_world_size(),
            shuffle=config.shuffle_train,
            drop_last=False,
            persistent_workers=config.train_loader_workers > 0,
        ),
        single_pass_length=config.single_pass_length,
    )
    return loader


def get_validation_loader(config: ClassificationConfig) -> DataLoader:
    return auto_dataloader(
        get_validation_dataset(config),
        num_workers=config.valid_loader_workers,
        batch_size=config.batch_size_valid * idist.get_world_size(),
        shuffle=False,
        drop_last=False,
        persistent_workers=config.valid_loader_workers > 0,
    )
