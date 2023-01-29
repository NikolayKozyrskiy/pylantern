from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, TypeVar, Tuple

from pydantic import Field
from torch import nn
from torchvision import models
from matches.loop import Loop

from ...config import ClassificationConfig
from ...models.qresnet_8x import ResNet18_8x

if TYPE_CHECKING:
    from .pipeline import QuantClassificationPipeline


class QuantClassificationConfig(ClassificationConfig):
    num_bits_a: int
    num_bits_w: int
    do_quantize_fc: bool = False

    def model(self) -> nn.Module:
        return ResNet18_8x(self.num_bits_a, self.num_classes)

    def resume(self, loop: Loop, pipeline: "QuantClassificationPipeline"):
        if self.resume_from_checkpoint is not None:
            loop.state_manager.read_state(
                self.resume_from_checkpoint,
                skip_keys=[
                    "scheduler",
                ],
            )

    def postprocess(self, loop: Loop, pipeline: "QuantClassificationPipeline"):
        pass
