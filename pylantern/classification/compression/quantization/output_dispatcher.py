from typing import Dict, List

import torch
from torch import Tensor
import torch.nn.functional as F
from torchtyping import TensorType

from pylantern.common.metrics import correct_labels
from ...output_dispatcher import OutputDispatcherClr
from ...pipeline import ClassificationPipeline


class QuantOutputDispatcherClr(OutputDispatcherClr):
    def __init__(
        self,
        loss_aggregation_weigths: Dict[str, float],
        metrics: List[str],
    ):
        super().__init__(loss_aggregation_weigths, metrics)
