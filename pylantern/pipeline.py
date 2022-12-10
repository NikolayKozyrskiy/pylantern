import contextlib
from typing import List, TYPE_CHECKING

from torch import Tensor
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

    @contextlib.contextmanager
    def batch_scope(self, batch: List[Tensor]):
        raise NotImplementedError()
