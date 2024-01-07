from contextlib import contextmanager
from typing import TYPE_CHECKING, List, NamedTuple, Optional

import torch
from matches.shortcuts.dag import graph_node
from torch import Tensor
from torch.nn import Module

from pylantern.pipeline import BasePipeline

if TYPE_CHECKING:
    from .config import ClassificationConfig
    from .data.dataset import ClassificationDatasetItem


class ClassifierOutput(NamedTuple):
    logits: Optional[Tensor]


class ClassificationPipeline(BasePipeline):
    def __init__(
        self,
        config: "ClassificationConfig",
        classifier_model: Module,
    ):
        BasePipeline.__init__(self, config=config)
        self.config = config
        self.classifier_model = classifier_model
        self.num_classes = config.num_classes

    @contextmanager
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
        return self.classifier_model(self.gt_images())

    @graph_node
    def predict_labels(self) -> Tensor:
        return self.predict_logits().max(1)[1]


def pipeline_from_config(
    config: "ClassificationConfig", device: str
) -> ClassificationPipeline:
    model = config.classifier_model().to(device)
    return ClassificationPipeline(config, classifier_model=model)
