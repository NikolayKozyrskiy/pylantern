from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from matches.shortcuts.dag import ComputationGraph

if TYPE_CHECKING:
    from .config import BaseConfig


class BasePipeline(ComputationGraph):
    def __init__(
        self,
        config: "BaseConfig",
    ):
        super().__init__()
        self.config = config
        self.batch = None

    @contextmanager
    def batch_scope(self, batch: Any):
        raise NotImplementedError()
