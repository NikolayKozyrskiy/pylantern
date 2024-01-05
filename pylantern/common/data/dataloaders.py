from typing import TYPE_CHECKING

import ignite.distributed as idist
from ignite.distributed.auto import auto_dataloader
from matches.loop.loader_scheduling import DataloaderSchedulerWrapper

if TYPE_CHECKING:
    from torch.utils.data import DataLoader, Dataset

    from pylantern.config import BaseConfig


def get_train_loader(
    config: "BaseConfig", dataset: "Dataset"
) -> DataloaderSchedulerWrapper:
    loader = DataloaderSchedulerWrapper(
        auto_dataloader(
            dataset,
            num_workers=config.train_loader_workers,
            batch_size=config.train_batch_size * idist.get_world_size(),
            shuffle=config.shuffle_train,
            drop_last=False,
            persistent_workers=config.train_loader_workers > 0,
        ),
        single_pass_length=config.single_pass_length,
    )

    return loader


def get_validation_loader(config: "BaseConfig", dataset: "Dataset") -> "DataLoader":
    return auto_dataloader(
        dataset,
        num_workers=config.valid_loader_workers,
        batch_size=config.valid_batch_size * idist.get_world_size(),
        shuffle=False,
        drop_last=False,
        persistent_workers=config.valid_loader_workers > 0,
    )


def get_eval_loader(config: "BaseConfig", dataset: "Dataset") -> "DataLoader":
    return auto_dataloader(
        dataset,
        num_workers=config.valid_loader_workers,
        batch_size=config.valid_batch_size * idist.get_world_size(),
        shuffle=False,
        drop_last=False,
        persistent_workers=config.valid_loader_workers > 0,
    )
