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
from torch.nn import Module
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

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

from .common.wandb import WandBLoggingSink

if TYPE_CHECKING:
    from .pipeline import Pipeline


C = TypeVar("C", bound=Callable)


class BaseConfig(BaseModel):
    data_root: str

    loss_aggregation_weigths: Dict[str, float]
    metrics: List[str]

    batch_size_train: int = 2
    batch_size_valid: int = 2
    lr: float = 1e-2
    max_epoch: int = 100
    train_transforms: list[Callable] = []
    valid_transforms: list[Callable] = []

    comment: Optional[str] = "default_comment"
    train_loader_workers: int = 4
    valid_loader_workers: int = 4
    single_pass_length: float = 1.0
    monitor: str = ""
    resume_from_checkpoint: Optional[Path] = None
    shuffle_train: bool = True

    output_config: list[Callable[["Pipeline", Executor, Path], None]] = []
    preview_image_fns: List[Callable] = []
    log_vis_fns: List[Callable] = []

    def optimizer(self, model: Module) -> Optimizer:
        return SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)

    def scheduler(self, optimizer: Optimizer) -> Optional[LRSchedulerProto]:
        return LRSchedulerWrapper(
            CosineAnnealingLR(optimizer, T_max=self.max_epoch),
            scope_type=SchedulerScopeType.BATCH,
        )

    def resume(self, loop: Loop, pipeline: "Pipeline") -> None:
        if self.resume_from_checkpoint is not None:
            loop.state_manager.read_state(
                self.resume_from_checkpoint,
                skip_keys=[
                    "scheduler",
                ],
            )

    def postprocess(self, loop: Loop, pipeline: "Pipeline") -> None:
        pass

    def train_callbacks(self, dev: bool, *args, **kwargs) -> List[Callback]:
        callbacks = [WandBLoggingSink(self.comment, self), TqdmProgressCallback()]
        if not dev:
            callbacks += [
                EnsureWorkdirCleanOrDevMode(),
                BestModelSaver(self.monitor, metric_mode="max", logdir_suffix=""),
                LastModelSaverCallback(),
            ]
        return callbacks

    def valid_callbacks(self, *args, **kwargs) -> List[Callback]:
        return [TqdmProgressCallback()]


def load_config(config_path: Path, desired_class):
    text = config_path.read_text()

    ctx = {}
    exec(text, ctx)

    config = ctx["config"]

    assert isinstance(config, desired_class), (config.__class__, desired_class)
    return config
