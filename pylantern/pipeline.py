from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterable, Mapping, Union

import torch
from matches.shortcuts.dag import ComputationGraph

if TYPE_CHECKING:
    from .config import BaseConfig


class BasePipeline(ComputationGraph):
    def __init__(
        self,
        config: "BaseConfig",
        device: Union[str, torch.device, None] = None,
    ):
        ComputationGraph.__init__(self)
        self.config = config
        self.device = device
        self.batch = None

    @contextmanager
    def batch_scope(self, batch: Union[Mapping, Iterable]):
        raise NotImplementedError()
