import os
import sys
from enum import Enum
from pathlib import Path
from shutil import copy
from typing import List, Optional

from matches.accelerators import DDPAccelerator, VanillaAccelerator
from matches.loop import Loop
from matches.utils import unique_logdir
import typer

from ..config import load_config, dump_config_json, dump_config_txt
from ..config_generator import ConfigGenerator, load_config_generator
from ..common.utils import (
    DevMode,
    load_pickle,
    dump_pickle,
    prepare_logdir,
    prepare_comment,
    copy_config,
    copy_config_generator,
)
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
    comment, config = prepare_comment(comment, config_path, config)
    logdir = prepare_logdir(logdir, comment)
    copy_config(config_path, logdir)

    loop = Loop(
        logdir,
        config.train_callbacks(dev_mode != dev_mode.DISABLED),
        loader_override=dev_mode.value,
    )
    loop.launch(
        train_fn,
        DDPAccelerator(gpus) if gpus is not None else VanillaAccelerator("cpu"),
        config=config,
    )


@app.command()
def train_replays(
    config_generator_path: Path,
    root_log_dir: Path,
    comment_postfix: str = typer.Option(None, "--comment-postfix", "-C"),
    gpus: str = typer.Option(None, "--gpus", "--gpu", "-g"),
    dev_mode: DevMode = typer.Option(DevMode.DISABLED, "--dev-mode", "-m"),
):
    root_log_dir.mkdir(exist_ok=True, parents=True)
    copy_config_generator(config_generator_path, root_log_dir)
    config_generator: ConfigGenerator = load_config_generator(config_generator_path)

    for idx, config in enumerate(config_generator()):
        comment = (
            f"{config.comment}__{comment_postfix}_genit_{idx}"
            if comment_postfix is not None
            else f"{config.comment}__genit_{idx}"
        )
        config.comment = comment

        logdir = root_log_dir / comment
        logdir.mkdir(exist_ok=True, parents=True)
        # dump_config_json(config, logdir / "config.json")
        dump_config_txt(config, logdir / "config")
        # dump_pickle(config, logdir / "config.pkl")

        loop = Loop(
            logdir,
            config.train_callbacks(dev_mode != dev_mode.DISABLED),
            loader_override=dev_mode.value,
        )

        loop.launch(
            train_fn,
            DDPAccelerator(gpus) if gpus is not None else VanillaAccelerator("cpu"),
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
        DDPAccelerator(gpus) if gpus is not None else VanillaAccelerator("cpu"),
        config=config,
        checkpoint=checkpoint,
        data_root=data_root,
        output_name=output_name,
    )


if __name__ == "__main__":
    app()
