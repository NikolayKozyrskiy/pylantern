from contextlib import contextmanager
from typing import TYPE_CHECKING, ClassVar, Dict, Iterable, Mapping, Protocol, Union

import torch
from matches.shortcuts.dag import ComputationGraph

if TYPE_CHECKING:
    from pylantern.inference.config import BaseInferenceConfig


class DataClass(Protocol):
    __dataclass_fields__: ClassVar[Dict]


class BaseInferencePipeline(ComputationGraph):
    def __init__(
        self,
        config: "BaseInferenceConfig",
        device: Union[str, torch.device, None] = None,
    ):
        ComputationGraph.__init__(self)
        self.config = config
        self.device = device
        self.data = None

    @contextmanager
    def data_item_scope(self, data: Union[Mapping, Iterable, DataClass]):
        raise NotImplementedError()
