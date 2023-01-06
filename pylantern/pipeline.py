from contextlib import contextmanager
from typing import Any, TYPE_CHECKING

from matches.shortcuts.dag import ComputationGraph

if TYPE_CHECKING:
    from .config import BaseConfig


class Pipeline(ComputationGraph):
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
