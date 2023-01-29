from pathlib import Path

import typer

from ..common.utils import DevMode
from ..main_routines import (
    train_routine,
    train_config_generator_routine,
    infer_routine,
    infer_config_generator_routine,
)
from .config import ClassificationConfig
from .compression.quantization.config import QuantClassificationConfig


app = typer.Typer()


@app.command()
def train(
    config_path: Path,
    comment: str = typer.Option(None, "--comment", "-C"),
    logdir: Path = typer.Option(None, "--logdir", "-l"),
    gpus: str = typer.Option(None, "--gpus", "--gpu", "-g"),
    dev_mode: DevMode = typer.Option(DevMode.DISABLED, "--dev-mode", "-m"),
):
    from .train_fns import train_fn

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
    gpus: str = typer.Option(None, "--gpus", "--gpu", "-g"),
    dev_mode: DevMode = typer.Option(DevMode.DISABLED, "--dev-mode", "-m"),
):
    from .train_fns import train_fn

    train_config_generator_routine(
        config_generator_path=config_generator_path,
        root_log_dir=root_log_dir,
        train_fn=train_fn,
        gpus=gpus,
        dev_mode=dev_mode,
    )


@app.command()
def infer(
    config_path: Path,
    logdir: Path = typer.Option(None, "--logdir", "-l"),
    checkpoint: str = typer.Option("best", "--checkpoint", "-c"),
    data_root: Path = typer.Option(None, "--data-root", "-d"),
    output_name: str = typer.Option(None, "--output-name", "-o"),
    gpus: str = typer.Option(None, "--gpus", "--gpu", "-g"),
):
    from .train_fns import infer_fn

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


@app.command()
def infer_config_generator(
    config_generator_path: Path,
    root_log_dir: Path = typer.Option(None, "--root-log-dir", "-l"),
    summary_path: Path = typer.Option(None, "--summary-path", "-s"),
    metric_name: str = typer.Option(None, "--metric-name", "-m"),
    checkpoint: str = typer.Option("best", "--checkpoint", "-c"),
    data_root: Path = typer.Option(None, "--data-root", "-d"),
    output_name: str = typer.Option(None, "--output-name", "-o"),
    gpus: str = typer.Option(None, "--gpus", "--gpu", "-g"),
):
    from .train_fns import infer_fn

    infer_config_generator_routine(
        config_generator_path=config_generator_path,
        infer_fn=infer_fn,
        root_log_dir=root_log_dir,
        summary_path=summary_path,
        metric_name=metric_name,
        checkpoint=checkpoint,
        data_root=data_root,
        output_name=output_name,
        gpus=gpus,
    )

@app.command()
def train_quantization(
    config_path: Path,
    comment: str = typer.Option(None, "--comment", "-C"),
    logdir: Path = typer.Option(None, "--logdir", "-l"),
    gpus: str = typer.Option(None, "--gpus", "--gpu", "-g"),
    dev_mode: DevMode = typer.Option(DevMode.DISABLED, "--dev-mode", "-m"),
):
    from .compression.quantization.train_fns import train_fn

    train_routine(
        config_path=config_path,
        config_cls=QuantClassificationConfig,
        train_fn=train_fn,
        comment=comment,
        logdir=logdir,
        gpus=gpus,
        dev_mode=dev_mode,
    )


@app.command()
def train_config_generator_quantization(
    config_generator_path: Path,
    root_log_dir: Path,
    gpus: str = typer.Option(None, "--gpus", "--gpu", "-g"),
    dev_mode: DevMode = typer.Option(DevMode.DISABLED, "--dev-mode", "-m"),
):
    from .compression.quantization.train_fns import train_fn

    train_config_generator_routine(
        config_generator_path=config_generator_path,
        root_log_dir=root_log_dir,
        train_fn=train_fn,
        gpus=gpus,
        dev_mode=dev_mode,
    )


@app.command()
def infer_quantization(
    config_path: Path,
    logdir: Path = typer.Option(None, "--logdir", "-l"),
    checkpoint: str = typer.Option("best", "--checkpoint", "-c"),
    data_root: Path = typer.Option(None, "--data-root", "-d"),
    output_name: str = typer.Option(None, "--output-name", "-o"),
    gpus: str = typer.Option(None, "--gpus", "--gpu", "-g"),
):
    from .compression.quantization.train_fns import infer_fn

    infer_routine(
        config_path=config_path,
        config_cls=QuantClassificationConfig,
        infer_fn=infer_fn,
        logdir=logdir,
        checkpoint=checkpoint,
        data_root=data_root,
        output_name=output_name,
        gpus=gpus,
    )


@app.command()
def infer_config_generator_quantization(
    config_generator_path: Path,
    root_log_dir: Path = typer.Option(None, "--root-log-dir", "-l"),
    summary_path: Path = typer.Option(None, "--summary-path", "-s"),
    metric_name: str = typer.Option(None, "--metric-name", "-m"),
    checkpoint: str = typer.Option("best", "--checkpoint", "-c"),
    data_root: Path = typer.Option(None, "--data-root", "-d"),
    output_name: str = typer.Option(None, "--output-name", "-o"),
    gpus: str = typer.Option(None, "--gpus", "--gpu", "-g"),
):
    from .compression.quantization.train_fns import infer_fn

    infer_config_generator_routine(
        config_generator_path=config_generator_path,
        infer_fn=infer_fn,
        root_log_dir=root_log_dir,
        summary_path=summary_path,
        metric_name=metric_name,
        checkpoint=checkpoint,
        data_root=data_root,
        output_name=output_name,
        gpus=gpus,
    )


if __name__ == "__main__":
    app()
