import json
import os
import pickle
import sys
from pathlib import Path
from shutil import copy
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
import tqdm.auto as tqdm
from ignite import distributed as idist
from matches.loop import Loop
from matches.utils import unique_logdir
from tqdm.contrib.concurrent import thread_map

if TYPE_CHECKING:
    from pylantern import BaseConfig


class attrdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


def load_cpt_full(
    loop: "Loop", config: "BaseConfig", skip_keys: Optional[Sequence[str]] = None
) -> None:
    loop.state_manager.read_state(config.checkpoint_path, skip_keys=skip_keys)
    return None


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


def execute_parallel(
    fn: Callable, *sequences: Sequence[Any], max_workers: int = 40, chunksize: int = 100
) -> Sequence[Any]:
    return thread_map(fn, *sequences, max_workers=max_workers, chunksize=chunksize)


def get_device() -> Union[str, torch.device]:
    if torch.cuda.is_available():
        if idist.get_world_size() > 1:
            return idist.device()
        else:
            return f"cuda:{torch.cuda.current_device()}"
    else:
        return "cpu"


def mkdir(dir_path: Union[Path, str, None]) -> Optional[Path]:
    if dir_path is not None:
        dir_path = Path(dir_path)
        dir_path.mkdir(exist_ok=True, parents=True)
        return dir_path
    else:
        return None


def assemble_dir(dir_path: Path) -> Path:
    dir_path = dir_path.resolve()
    dir_path.mkdir(exist_ok=True, parents=True)
    return dir_path


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


def enumerate_normalized(iterable: Iterable, len: int):
    for i, e in enumerate(iterable):
        yield i / len, e


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
    obj: Dict[str, Any],
    file_path: Union[Path, str],
    indent: int = 2,
    mode: str = "w",
) -> None:
    _file_path = str(file_path)
    if not _file_path.endswith(".json"):
        _file_path += ".json"
    with open(_file_path, mode) as f:
        json.dump(obj, f, indent=indent)
    return None


def append_json(
    obj: Dict[str, Any],
    file_path: Union[Path, str],
    merge_fn: Callable,
    indent: int = 2,
) -> None:
    _file_path = str(file_path)
    if not _file_path.endswith(".json"):
        _file_path += ".json"
    if os.path.exists(_file_path):
        with open(_file_path, "r") as f:
            f_data: dict = json.load(f)
        merge_fn(f_data, obj)
    else:
        f_data = obj
    with open(_file_path, "w") as f:
        json.dump(f_data, f, indent=indent)
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
