from typing import Dict, List

import torch
from torch import Tensor
import torch.nn.functional as F
from torchtyping import TensorType

from ..common.metrics import correct_labels
from ..output_dispatcher import OutputDispatcher
from .pipeline import ClassificationPipeline


class OutputDispatcherClr(OutputDispatcher):
    def __init__(
        self,
        loss_aggregation_weigths: Dict[str, float],
        metrics: List[str],
    ):
        super().__init__(loss_aggregation_weigths, metrics)

    @staticmethod
    def clr__cross_entropy(
        pipeline: ClassificationPipeline,
    ) -> TensorType["batch_size"]:
        pred = pipeline.predict_labels()
        gt = pipeline.gt_labels()
        return F.cross_entropy(pred, gt, reduction="none")

    @staticmethod
    def clr__accuracy(pipeline: ClassificationPipeline) -> TensorType["batch_size"]:
        pred = pipeline.predict_labels()
        gt = pipeline.gt_labels()
        return correct_labels(pred, gt)
