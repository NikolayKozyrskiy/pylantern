from concurrent.futures import Executor
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
)

from pydantic import BaseModel, Field
from torch import nn
from torch.optim import Optimizer, SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
import torchvision.transforms as tr

from matches.loop import Loop
from matches.shortcuts.optimizer import (
    LRSchedulerProto,
    LRSchedulerWrapper,
    SchedulerScopeType,
)
from matches.callbacks import (
    Callback,
    BestModelSaver,
    TqdmProgressCallback,
    LastModelSaverCallback,
    EnsureWorkdirCleanOrDevMode,
)

from pylantern.common.wandb import WandBLoggingSink
from pylantern.classification.config import (
    ClassificationConfig,
    ClassificationDatasetName,
)
from pylantern.classification.pipeline import ClassificationPipeline
from pylantern.classification.transforms import train_basic_augs
from pylantern.classification.vis import log_to_wandb_gt_pred_labels
from pylantern.classification.models.resnet_small import resnet18


C = TypeVar("C", bound=Callable)


class Config(ClassificationConfig):
    def model(self) -> nn.Module:
        return models.resnet18(pretrained=True, progress=True)

    def resume(self, loop: Loop, pipeline: "ClassificationPipeline"):
        if self.resume_from_checkpoint is not None:
            loop.state_manager.read_state(
                self.resume_from_checkpoint,
                skip_keys=[
                    "scheduler",
                ],
            )

    def optimizer(self, model: nn.Module) -> Optimizer:
        return Adam(model.parameters(), lr=self.lr, weight_decay=1e-6)

    def scheduler(self, optimizer: Optimizer) -> Optional[LRSchedulerProto]:
        return LRSchedulerWrapper(
            CosineAnnealingLR(optimizer, T_max=self.max_epoch),
            scope_type=SchedulerScopeType.EPOCH,
        )

    def postprocess(self, loop: Loop, pipeline: "ClassificationPipeline") -> None:
        pass

    def train_callbacks(self, dev: bool, *args, **kwargs) -> List[Callback]:
        callbacks = [WandBLoggingSink(self.comment, self), TqdmProgressCallback()]
        if not dev:
            callbacks += [
                # EnsureWorkdirCleanOrDevMode(),
                BestModelSaver(self.monitor, metric_mode="max", logdir_suffix=""),
                LastModelSaverCallback(),
            ]
        return callbacks

    def valid_callbacks(self, *args, **kwargs) -> List[Callback]:
        return [TqdmProgressCallback()]


config = Config(
    data_root="_data/ILSVRC-12",
    num_classes=1000,
    dataset_name=ClassificationDatasetName.IMAGENET,
    image_hw=(256, 256),
    loss_aggregation_weigths={"clr/cross_entropy": 1.0},
    metrics=["clr/accuracy"],
    monitor="valid/clr/accuracy",
    batch_size_train=128,
    batch_size_valid=200,
    lr=1e-3,
    max_epoch=100,
    train_transforms=[
        tr.RandomResizedCrop(224),
        tr.RandomHorizontalFlip(),
        tr.RandomRotation(30),
        tr.ColorJitter(0.2, 0.2, 0.2, 0.2),
    ],
    valid_transforms=[tr.Resize(256), tr.CenterCrop(224)],
    comment="imagenet_resnet18",
    train_loader_workers=8,
    valid_loader_workers=10,
    single_pass_length=1.0,
    resume_from_checkpoint=None,
    shuffle_train=True,
    output_config=[],
    preview_image_fns=[],
    log_vis_fns=[],
)
