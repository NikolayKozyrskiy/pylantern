from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, TypeVar

from matches.callbacks import (
    BestMetricsReporter,
    BestModelSaver,
    Callback,
    EnsureWorkdirCleanOrDevMode,
    LastModelSaverCallback,
    TqdmProgressCallback,
    WandBLoggingSink,
)
from matches.loop import Loop
from matches.shortcuts.optimizer import LRSchedulerWrapper, SchedulerScopeType
from torch import nn
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models

from pylantern.model_zoo.models import resnet18_small
from pylantern.output_dispatcher import CriterionAggregation
from pylantern.tasks.classification.config import (
    ClassificationConfig,
    ClassificationDatasetName,
)
from pylantern.tasks.classification.models.vanilla_cnn import VanillaClassifier
from pylantern.tasks.classification.pipeline import ClassificationPipeline
from pylantern.tasks.classification.transforms import train_basic_augs
from pylantern.tasks.classification.visualization import log_to_wandb_gt_pred_labels

C = TypeVar("C", bound=Callable)


class Config(ClassificationConfig):
    def classifier_model(self) -> nn.Module:
        return resnet18_small(num_classes=self.num_classes, requires_grad=True)
        # return VanillaClassifier(num_classes=self.num_classes)

    def resume(self, loop: Loop, pipeline: "ClassificationPipeline"):
        # if self.checkpoint_path is not None:
        #     loop.state_manager.read_state(
        #         self.checkpoint_path,
        #         skip_keys=[
        #             "scheduler",
        #         ],
        #     )
        pass

    def optimizer(self, model: nn.Module) -> Optimizer:
        return SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)

    def scheduler(self, optimizer: Optimizer) -> LRSchedulerWrapper:
        return LRSchedulerWrapper(
            CosineAnnealingLR(optimizer, T_max=self.max_epoch),
            scope_type=SchedulerScopeType.EPOCH,
        )

    def postprocess(self, loop: Loop, pipeline: "ClassificationPipeline") -> None:
        pass

    def train_callbacks(self, dev: bool, *args, **kwargs) -> List[Callback]:
        callbacks = [
            WandBLoggingSink(self.comment, self.dict()),
            TqdmProgressCallback(),
        ]
        if not dev:
            callbacks += [
                # EnsureWorkdirCleanOrDevMode(),
                BestModelSaver(self.monitor, metric_mode="max", logdir_suffix=""),
                LastModelSaverCallback(),
                BestMetricsReporter(
                    metrics_name_mode={
                        self.monitor: "max",
                        "valid/clr/cross_entropy": "min",
                    }
                ),
            ]
        return callbacks

    def valid_callbacks(self, *args, **kwargs) -> List[Callback]:
        return [TqdmProgressCallback()]


config = Config(
    root_path="_d",
    comment="cifar10_resnet18",
    num_classes=10,
    dataset_name=ClassificationDatasetName.CIFAR10,
    image_hw=(32, 32),
    criterion_aggregation=CriterionAggregation(default={"clr/cross_entropy": 1.0}),
    metrics=["clr/accuracy"],
    monitor="valid/clr/accuracy",
    train_batch_size=200,
    valid_batch_size=250,
    shuffle_train=True,
    lr=1e-1,
    max_epoch=10,
    train_transforms=train_basic_augs(crop_size=(32, 32)),
    valid_transforms=[],
    train_loader_workers=8,
    valid_loader_workers=8,
    single_pass_length=1.0,
    checkpoint_path=None,
    output_config=[],
    preview_config=[],
    log_vis_fns=[],  # [log_to_wandb_gt_pred_labels]
)
