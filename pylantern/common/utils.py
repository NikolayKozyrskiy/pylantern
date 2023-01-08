from enum import Enum
import json
from typing import (
    Dict,
    Iterable,
    Tuple,
    Any,
    Union,
    List,
    Optional,
    TYPE_CHECKING,
    Type,
    Generator,
)
from pathlib import Path
import pickle
from shutil import copy
import sys

import numpy as np
from ignite.metrics.accumulation import Average
import pandas as pd
import torch
from torch.optim import Optimizer
import tqdm.auto as tqdm
from matches.loop import IterationType, Loop
from matches.utils import unique_logdir
from matches.shortcuts.callbacks import get_metrics_summary

if TYPE_CHECKING:
    from ..config import BaseConfig


class DevMode(str, Enum):
    DISABLED = "disabled"
    SHORT = "short"
    OVERFIT_BATCH = "overfit-batch"


def wrap_tqdm(
    iterable: Iterable[Any], name: str, length: int, leave: bool = True
) -> Generator[Any, None, None]:
    progress_meter = tqdm.tqdm(desc=name, file=sys.stderr, leave=leave)
    try:
        for item in iterable:
            if progress_meter.total != length:
                progress_meter.reset(total=length)
            yield item
            progress_meter.update(1)
    except GeneratorExit:
        progress_meter.close()
        raise
    progress_meter.close()


def get_device() -> str:
    device = (
        f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    )
    return device


def prepare_comment(
    comment: Optional[str], config_path: Union[Path, str], config: "BaseConfig"
) -> Tuple[str, "BaseConfig"]:
    comment = comment or config.comment
    if comment is None:
        comment = Path(config_path).stem
    config.comment = comment
    return comment, config


def prepare_logdir(logdir: Optional[Path], comment: str) -> Path:
    logdir = logdir or unique_logdir(Path("logs/"), comment)
    logdir.mkdir(exist_ok=True, parents=True)
    return logdir


def copy_config(config_path: Union[Path, str], logdir: Path) -> None:
    copy(config_path, logdir / "config.py", follow_symlinks=True)


def copy_config_generator(
    config_generator_path: Union[Path, str], root_log_dir: Path
) -> None:
    copy(
        config_generator_path,
        root_log_dir / "config_generator.py",
        follow_symlinks=True,
    )


def print_best_metrics_summary(loop: Loop) -> None:
    summary = get_metrics_summary(loop)
    if summary is not None:
        print(f"Metrics summary:\n{pd.DataFrame(summary)}")
    return None


def log_optimizer_lrs(
    loop: Loop,
    optimizer: Optimizer,
    prefix: str = "lr",
    iteration: IterationType = IterationType.AUTO,
) -> None:
    for i, g in enumerate(optimizer.param_groups):
        loop.metrics.log(f"{prefix}/group_{i}", g["lr"], iteration)


def consume_metric(loop: Loop, avg_dict: Dict[str, Average], prefix: str) -> None:
    for name, value in avg_dict.items():
        loop.metrics.consume(f"{prefix}/{name}", value)


def enumerate_normalized(iterable: Iterable, len: int):
    for i, e in enumerate(iterable):
        yield i / len, e


def tensor_to_image(
    tensor: torch.Tensor,
    val_range: Tuple[float, float] = (0.0, 1.0),
    keepdim: bool = False,
) -> "np.ndarray":
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(tensor)}")

    if len(tensor.shape) > 4 or len(tensor.shape) < 2:
        raise ValueError("Input size must be a two, three or four dimensional tensor")

    input_shape = tensor.shape
    image: "np.ndarray" = tensor.cpu().detach().numpy()

    if len(input_shape) == 2:
        # (H, W) -> (H, W)
        pass
    elif len(input_shape) == 3:
        # (C, H, W) -> (H, W, C)
        if input_shape[0] == 1:
            # Grayscale for proper plt.imshow needs to be (H,W)
            image = image.squeeze()
        else:
            image = image.transpose(1, 2, 0)
    elif len(input_shape) == 4:
        # (B, C, H, W) -> (B, H, W, C)
        image = image.transpose(0, 2, 3, 1)
        if input_shape[0] == 1 and not keepdim:
            image = image.squeeze(0)
        if input_shape[1] == 1:
            image = image.squeeze(-1)
    else:
        raise ValueError(f"Cannot process tensor with shape {input_shape}")

    if val_range[0] == -1.0 and val_range[1] == 1.0:
        image = image / 2.0 + 0.5

    image = (image * 255).clip(0, 255).astype(np.uint8)
    return image


def dump_pickle(obj: Any, file_path: Union[Path, str]) -> None:
    _file_path = str(file_path)
    if not _file_path.endswith(".pkl"):
        _file_path += ".pkl"
    with open(_file_path, "wb") as f:
        pickle.dump(obj, f)
    return None


def load_pickle(file_path: Union[Path, str]) -> Any:
    _file_path = str(file_path)
    if not _file_path.endswith(".pkl"):
        _file_path += ".pkl"
    with open(_file_path, "rb") as f:
        obj = pickle.load(f)
    return obj


def dump_json(
    obj: Dict[str, Any], file_path: Union[Path, str], indent: int = 2
) -> None:
    _file_path = str(file_path)
    if not _file_path.endswith(".json"):
        _file_path += ".json"
    with open(_file_path, "w") as f:
        json.dump(obj, f, indent=indent)
    return None


def load_json(file_path: Union[Path, str]) -> Dict[Any, Any]:
    _file_path = str(file_path)
    if not _file_path.endswith(".json"):
        _file_path += ".json"
    with open(_file_path, "r") as f:
        res = json.load(f)
    return res


def dump_txt(obj: Any, file_path: Union[Path, str]) -> None:
    _file_path = str(file_path)
    if not _file_path.endswith(".txt"):
        _file_path += ".txt"
    with open(_file_path, "w") as f:
        f.write(str(obj))
    return None
