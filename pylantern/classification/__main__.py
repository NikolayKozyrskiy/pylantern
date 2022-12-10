import os
import sys
from enum import Enum
from pathlib import Path
from shutil import copy
from typing import List, Optional

from matches.accelerators import DDPAccelerator
from matches.loop import Loop
from matches.utils import unique_logdir
import typer

from ..config import load_config
from ..common.utils import DevMode
from .config import ClassificationConfig
from .train_fns import train_fn, infer_fn


app = typer.Typer()


@app.command()
def train(
    config_path: Path,
    comment: str = typer.Option(None, "--comment", "-C"),
    logdir: Path = None,
    gpus: str = typer.Option(None, "--gpus", "--gpu", "-g"),
    dev_mode: DevMode = typer.Option(DevMode.DISABLED, "--dev-mode", "-m"),
):
    config: ClassificationConfig = load_config(config_path, ClassificationConfig)

    comment = comment or config.comment
    if comment is None:
        comment = config_path.stem
    config.comment = comment

    logdir = logdir or unique_logdir(Path("logs/"), comment)
    logdir.mkdir(exist_ok=True, parents=True)

    copy(config_path, logdir / "config.py", follow_symlinks=True)
    loop = Loop(
        logdir,
        config.train_callbacks(dev_mode != dev_mode.DISABLED),
        loader_override=dev_mode.value,
    )

    loop.launch(
        train_fn,
        DDPAccelerator(gpus),
        config=config,
    )


@app.command()
def infer(
    config_path: Path,
    logdir: Path,
    checkpoint: str = typer.Option("best", "-c"),
    gpus: str = typer.Option(None, "--gpus", "--gpu", "-g"),
    data_root: Optional[Path] = None,
    output_name: Optional[str] = None,
):
    config: ClassificationConfig = load_config(config_path, ClassificationConfig)
    loop = Loop(
        logdir,
        config.valid_callbacks(),
    )

    loop.launch(
        infer_fn,
        DDPAccelerator(gpus),
        config=config,
        checkpoint=checkpoint,
        data_root=data_root,
        output_name=output_name,
    )


if __name__ == "__main__":
    app()
