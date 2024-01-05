from types import ModuleType
from typing import TYPE_CHECKING, Dict, List, Optional

import torch
import torch.nn.functional as F
from matches.loop import Loop
from torch import Tensor

from pylantern.common.metrics import correct_labels
from pylantern.output_dispatcher import BaseOutputDispatcher

if TYPE_CHECKING:
    from .config import ClassificationConfig
    from .pipeline import ClassificationPipeline


class OutputDispatcherClr(BaseOutputDispatcher):
    def __init__(
        self,
        config: "ClassificationConfig",
        complex_criterions_module: Optional[ModuleType],
        *args,
        **kwargs,
    ):
        super().__init__(
            config=config,
            complex_criterions_module=complex_criterions_module,
            *args,
            **kwargs,
        )

    def clr__cross_entropy(
        self, pipeline: "ClassificationPipeline", loop: "Loop", *args, **kwargs
    ) -> Tensor:
        pred = pipeline.predict_logits()
        gt = pipeline.gt_labels()
        return F.cross_entropy(pred, gt, reduction="mean")

    def clr__accuracy(
        self, pipeline: "ClassificationPipeline", loop: "Loop", *args, **kwargs
    ) -> Tensor:
        pred = pipeline.predict_logits()
        gt = pipeline.gt_labels()
        return correct_labels(pred, gt).mean()
