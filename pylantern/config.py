from concurrent.futures import Executor
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, TypeVar, Union

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
    WandBLoggingSink,
    BestMetricsReporter,
)

from .common.utils import load_pickle, dump_json, dump_txt

if TYPE_CHECKING:
    from .pipeline import Pipeline


C = TypeVar("C", bound=Callable)


class BaseConfig(BaseModel):
    data_root: str

    loss_aggregation_weigths: Dict[str, float]
    metrics: List[str]
    monitor: str

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
        callbacks = [
            WandBLoggingSink(self.comment, get_config_dict(self)),
            TqdmProgressCallback(),
        ]
        if not dev:
            callbacks += [
                EnsureWorkdirCleanOrDevMode(),
                BestModelSaver(self.monitor, metric_mode="max", logdir_suffix=""),
                LastModelSaverCallback(),
                BestMetricsReporter(
                    metrics_name_mode={
                        self.monitor: "max",
                    }
                ),
            ]
        return callbacks

    def valid_callbacks(self, *args, **kwargs) -> List[Callback]:
        return [TqdmProgressCallback()]


def get_config_dict(config: BaseConfig) -> Dict[str, str]:
    return {k: str(v) for k, v in config.dict().items()}


def dump_config_dict(config: BaseConfig, save_path: Union[Path, str]) -> None:
    dump_json(get_config_dict(config), save_path, indent=2)
    return None


def load_config(config_path: Union[Path, str], desired_class):
    if str(config_path).endswith(".py"):
        config = _load_config_from_py(config_path)
    elif str(config_path).endswith(".pkl"):
        config = load_pickle(config_path)
    else:
        raise ValueError(f"Given config file {config_path.stem} is not supported!")

    assert isinstance(config, desired_class), (config.__class__, desired_class)
    return config


def _load_config_from_py(config_path: Path):
    text = config_path.read_text()
    ctx = {}
    exec(text, ctx)
    return ctx["config"]


if __name__ == "__main__":
    conf = BaseConfig(
        data_root="_a", loss_aggregation_weigths={"l": 1.0}, metrics=["m"]
    )
    print(conf.json(indent=2))
    print("=" * 100)
    print(conf.dict())
