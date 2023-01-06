from pathlib import Path

import typer

from ..common.utils import DevMode
from ..main_routines import (
    train_routine,
    train_config_generator_routine,
    infer_routine,
)
from .config import ClassificationConfig
from .train_fns import train_fn, infer_fn


app = typer.Typer()


@app.command()
def train(
    config_path: Path,
    comment: str = typer.Option(None, "--comment", "-C"),
    logdir: Path = typer.Option(None, "--logdir", "-l"),
    gpus: str = typer.Option(None, "--gpus", "--gpu", "-g"),
    dev_mode: DevMode = typer.Option(DevMode.DISABLED, "--dev-mode", "-m"),
):
    train_routine(
        config_path=config_path,
        config_cls=ClassificationConfig,
        train_fn=train_fn,
        comment=comment,
        logdir=logdir,
        gpus=gpus,
        dev_mode=dev_mode,
    )


@app.command()
def train_config_generator(
    config_generator_path: Path,
    root_log_dir: Path,
    comment_postfix: str = typer.Option(None, "--comment-postfix", "-C"),
    gpus: str = typer.Option(None, "--gpus", "--gpu", "-g"),
    dev_mode: DevMode = typer.Option(DevMode.DISABLED, "--dev-mode", "-m"),
):
    train_config_generator_routine(
        config_generator_path=config_generator_path,
        root_log_dir=root_log_dir,
        train_fn=train_fn,
        comment_postfix=comment_postfix,
        gpus=gpus,
        dev_mode=dev_mode,
    )


@app.command()
def infer(
    config_path: Path,
    logdir: Path = typer.Option(None, "--logdir", "-l"),
    checkpoint: str = typer.Option("best", "--checkpoint", "-c"),
    gpus: str = typer.Option(None, "--gpus", "--gpu", "-g"),
    data_root: Path = typer.Option(None, "--data-root", "-d"),
    output_name: str = typer.Option(None, "--output-name", "-o"),
):
    infer_routine(
        config_path=config_path,
        config_cls=ClassificationConfig,
        infer_fn=infer_fn,
        logdir=logdir,
        checkpoint=checkpoint,
        data_root=data_root,
        output_name=output_name,
        gpus=gpus,
    )


if __name__ == "__main__":
    app()
