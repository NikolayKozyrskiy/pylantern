import contextlib
from typing import Union, List, Tuple, Type, Dict, Optional, NamedTuple, TYPE_CHECKING

import torch
from torch import Tensor
import torch.nn.functional as F
from matches.shortcuts.dag import ComputationGraph, graph_node

# from .models import AE, MPL, AEOutput
# from .config import DatasetName, TrainSetup, AEArchitecture
from ..pipeline import Pipeline

# if TYPE_CHECKING:
#     from .config import EClrConfig


class ClassifierOutput(NamedTuple):
    logits: Optional[Tensor]


class ClassificationPipeline(Pipeline):
    def __init__(
        self,
        config: "EClrConfig",
        model: Module,
        classes_num: int,
    ):
        super().__init__(config)
        self.config = config
        self.model = model
        self.classes_num = classes_num
        self.batch = None

    @contextlib.contextmanager
    def batch_scope(self, batch: List[Tensor]):
        try:
            with self.cache_scope():
                self.batch = process_batch(batch)
                yield
        finally:
            self.batch = None