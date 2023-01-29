from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
)

from torch import nn
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models

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
    WandBLoggingSink,
    BestMetricsReporter,
)

from pylantern.classification.config import ClassificationDatasetName
from pylantern.classification.compression.quantization.config import (
    QuantClassificationConfig,
)
from pylantern.classification.compression.quantization.pipeline import (
    QuantClassificationPipeline,
)
from pylantern.classification.transforms import train_basic_augs
from pylantern.classification.vis import log_to_wandb_gt_pred_labels
from pylantern.classification.models.qresnet_8x import ResNet18_8x

C = TypeVar("C", bound=Callable)


class Config(QuantClassificationConfig):
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

    def optimizer(self, model: nn.Module) -> Optimizer:
        return SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)

    def scheduler(self, optimizer: Optimizer) -> Optional[LRSchedulerProto]:
        return LRSchedulerWrapper(
            CosineAnnealingLR(optimizer, T_max=self.max_epoch),
            scope_type=SchedulerScopeType.EPOCH,
        )

    def postprocess(self, loop: Loop, pipeline: "QuantClassificationPipeline") -> None:
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
    num_bits_a=8,
    num_bits_w=8,
    do_quantize_fc=False,
    data_root="_data",
    num_classes=10,
    dataset_name=ClassificationDatasetName.CIFAR10,
    image_hw=(32, 32),
    loss_aggregation_weigths={"clr/cross_entropy": 1.0},
    metrics=["clr/accuracy"],
    monitor="valid/clr/accuracy",
    batch_size_train=200,
    batch_size_valid=250,
    lr=1e-1,
    max_epoch=10,
    train_transforms=train_basic_augs(crop_size=(32, 32)),
    valid_transforms=[],
    comment="quant_cifar10_qresnet18_8w8a",
    train_loader_workers=8,
    valid_loader_workers=8,
    single_pass_length=0.2,
    resume_from_checkpoint=None,
    shuffle_train=True,
    output_config=[],
    preview_image_fns=[],
    log_vis_fns=[],  # [log_to_wandb_gt_pred_labels]
)
