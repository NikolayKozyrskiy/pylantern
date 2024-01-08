from __future__ import annotations

import functools
from concurrent.futures import Executor
from pathlib import Path
from typing import TYPE_CHECKING, Callable, TypeVar

from typing_extensions import Concatenate, ParamSpec

if TYPE_CHECKING:
    from pylantern.pipeline import BasePipeline

Args = ParamSpec("Args")
R = TypeVar("R")


def delayed(
    f: Callable[Concatenate["BasePipeline", Executor, Path, Args], R]
) -> Callable[[Args], Callable[["BasePipeline", Executor, Path], R]]:
    @functools.wraps(f)
    def partial(*args: Args.args, **kwargs: Args.kwargs):
        return functools.partial(f, *args, **kwargs)

    # noinspection PyTypeChecker
    return partial
