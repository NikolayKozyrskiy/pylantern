from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
)

from pydantic import Field
from torch import nn
from torchvision import models
from matches.loop import Loop

from ..config import Config

if TYPE_CHECKING:
    from .pipeline import ClassificationPipeline


C = TypeVar("C", bound=Callable)


class ClassificationDatasetName(str, Enum):
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"
    IMAGENET = "imagenet"


class ClassificationConfig(Config):
    num_classes: int
    dataset_name: ClassificationDatasetName = ClassificationDatasetName.CIFAR10

    def model(self) -> nn.Module:
        model = models.mobilenet_v2(pretrained=True, progress=True)
        model.classifier[1] = nn.Linear(model.last_channel, self.num_classes)
        return model

    def resume(self, loop: Loop, pipeline: "ClassificationPipeline"):
        if self.resume_from_checkpoint is not None:
            loop.state_manager.read_state(
                self.resume_from_checkpoint,
                skip_keys=[
                    "scheduler",
                ],
            )

    def postprocess(self, loop: Loop, pipeline: "ClassificationPipeline"):
        pass
