import contextlib
from typing import List, Optional, NamedTuple, TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import Module
from matches.shortcuts.dag import graph_node

from ..pipeline import Pipeline

if TYPE_CHECKING:
    from .config import ClassificationConfig
    from .data.dataset import ClassificationDatasetItem


class ClassifierOutput(NamedTuple):
    logits: Optional[Tensor]


class ClassificationPipeline(Pipeline):
    def __init__(
        self,
        config: "ClassificationConfig",
        model: Module,
    ):
        super().__init__(config)
        self.config = config
        self.model = model
        self.num_classes = config.num_classes

    @contextlib.contextmanager
    def batch_scope(
        self, batch: "ClassificationDatasetItem"
    ):  # TODO: refactor batch here
        try:
            with self.cache_scope():
                self.batch = batch
                yield
        finally:
            self.batch = None

    @graph_node
    def gt_images(self) -> Tensor:
        img = self.batch.image
        return img

    @graph_node
    def gt_labels(self) -> Tensor:
        label = self.batch.label
        return label

    @graph_node
    def predict_logits(self) -> Tensor:
        return self.model(self.gt_images())

    @graph_node
    def predict_labels(self) -> Tensor:
        return self.predict_logits().max(1)[1]


def pipeline_from_config(
    config: "ClassificationConfig", device: str
) -> ClassificationPipeline:
    model = config.model().to(device)
    return ClassificationPipeline(config, model=model)
