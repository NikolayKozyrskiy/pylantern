from pathlib import Path
from typing import List, Optional

import typer

from pylantern.common.train_utils import DevMode
from pylantern.main_routines import test_routine, train_routine
from pylantern.tasks.gan.pix2pix.configs import (
    BasePix2PixConfig,
    FaceGFPGANConfig,
    FacePix2PixHDConfig,
    FaceSpadeConfig,
    GFPGANConfig,
    Pix2PixHDConfig,
)
from pylantern.tasks.gan.pix2pix.train_loops.face import (
    test_face_gfpgan_fn,
    test_face_pix2pixhd_fn,
    test_face_spade_fn,
    train_face_gfpgan_fn,
    train_face_pix2pixhd_fn,
    train_face_spade_fn,
)
from pylantern.tasks.gan.pix2pix.train_loops.general import (
    test_baseline_pix2pix_fn,
    test_gfpgan_fn,
    test_pix2pixhd_fn,
    train_baseline_pix2pix_fn,
    train_gfpgan_fn,
    train_pix2pixhd_fn,
)

app = typer.Typer()


@app.command()
def run_train_baseline_pix2pix(
    config_path: Path = typer.Option(Path("./"), "-c", "--config-path"),
    comment: str = typer.Option(None, "--comment", "-C"),
    logdir: Path = typer.Option(None, "--logdir", "-l"),
    gpus: str = typer.Option(None, "--gpus", "--gpu", "-g"),
    dev_mode: DevMode = typer.Option(DevMode.DISABLED, "--dev-mode", "-m"),
):
    train_routine(
        config_path=config_path,
        config_cls=BasePix2PixConfig,
        train_fn=train_baseline_pix2pix_fn,
        comment=comment,
        logdir=logdir,
        gpus=gpus,
        dev_mode=dev_mode,
    )


@app.command()
def run_test_baseline_pix2pix(
    config_path: Path = typer.Option(Path("./"), "-c", "--config-path"),
    logdir: Path = typer.Option(None, "--logdir", "-l"),
    checkpoint: str = typer.Option("best", "-cpt", "--checkpoint"),
    data_root: Path = typer.Option(None, "--data-root", "-d"),
    output_name: str = typer.Option(None, "--output-name", "-o"),
    gpus: str = typer.Option(None, "--gpus", "--gpu", "-g"),
):
    test_routine(
        config_path=config_path,
        config_cls=BasePix2PixConfig,
        infer_fn=test_baseline_pix2pix_fn,
        logdir=logdir,
        checkpoint=checkpoint,
        data_root=data_root,
        output_name=output_name,
        gpus=gpus,
    )


@app.command()
def run_train_gfpgan(
    config_path: Path = typer.Option(Path("./"), "-c", "--config-path"),
    comment: str = typer.Option(None, "--comment", "-C"),
    logdir: Path = typer.Option(None, "--logdir", "-l"),
    gpus: str = typer.Option(None, "--gpus", "--gpu", "-g"),
    dev_mode: DevMode = typer.Option(DevMode.DISABLED, "--dev-mode", "-m"),
):
    train_routine(
        config_path=config_path,
        config_cls=GFPGANConfig,
        train_fn=train_gfpgan_fn,
        comment=comment,
        logdir=logdir,
        gpus=gpus,
        dev_mode=dev_mode,
    )


@app.command()
def run_test_gfpgan(
    config_path: Path = typer.Option(Path("./"), "-c", "--config-path"),
    logdir: Path = typer.Option(None, "--logdir", "-l"),
    checkpoint: str = typer.Option("best", "-cpt", "--checkpoint"),
    data_root: Path = typer.Option(None, "--data-root", "-d"),
    output_name: str = typer.Option(None, "--output-name", "-o"),
    gpus: str = typer.Option(None, "--gpus", "--gpu", "-g"),
):
    test_routine(
        config_path=config_path,
        config_cls=GFPGANConfig,
        infer_fn=test_gfpgan_fn,
        logdir=logdir,
        checkpoint=checkpoint,
        data_root=data_root,
        output_name=output_name,
        gpus=gpus,
    )


@app.command()
def run_train_pix2pixhd(
    config_path: Path = typer.Option(Path("./"), "-c", "--config-path"),
    comment: str = typer.Option(None, "--comment", "-C"),
    logdir: Path = typer.Option(None, "--logdir", "-l"),
    gpus: str = typer.Option(None, "--gpus", "--gpu", "-g"),
    dev_mode: DevMode = typer.Option(DevMode.DISABLED, "--dev-mode", "-m"),
):
    train_routine(
        config_path=config_path,
        config_cls=Pix2PixHDConfig,
        train_fn=train_pix2pixhd_fn,
        comment=comment,
        logdir=logdir,
        gpus=gpus,
        dev_mode=dev_mode,
    )


@app.command()
def run_test_pix2pixhd(
    config_path: Path = typer.Option(Path("./"), "-c", "--config-path"),
    logdir: Path = typer.Option(None, "--logdir", "-l"),
    checkpoint: str = typer.Option("best", "-cpt", "--checkpoint"),
    data_root: Path = typer.Option(None, "--data-root", "-d"),
    output_name: str = typer.Option(None, "--output-name", "-o"),
    gpus: str = typer.Option(None, "--gpus", "--gpu", "-g"),
):
    test_routine(
        config_path=config_path,
        config_cls=Pix2PixHDConfig,
        infer_fn=test_pix2pixhd_fn,
        logdir=logdir,
        checkpoint=checkpoint,
        data_root=data_root,
        output_name=output_name,
        gpus=gpus,
    )


@app.command()
def run_train_face_gfpgan(
    config_path: Path = typer.Option(Path("./"), "-c", "--config-path"),
    comment: str = typer.Option(None, "--comment", "-C"),
    logdir: Path = typer.Option(None, "--logdir", "-l"),
    gpus: str = typer.Option(None, "--gpus", "--gpu", "-g"),
    dev_mode: DevMode = typer.Option(DevMode.DISABLED, "--dev-mode", "-m"),
):
    train_routine(
        config_path=config_path,
        config_cls=FaceGFPGANConfig,
        train_fn=train_face_gfpgan_fn,
        comment=comment,
        logdir=logdir,
        gpus=gpus,
        dev_mode=dev_mode,
    )


@app.command()
def run_test_face_gfpgan(
    config_path: Path = typer.Option(Path("./"), "-c", "--config-path"),
    logdir: Path = typer.Option(None, "--logdir", "-l"),
    checkpoint: str = typer.Option("best", "-cpt", "--checkpoint"),
    data_root: Path = typer.Option(None, "--data-root", "-d"),
    output_name: str = typer.Option(None, "--output-name", "-o"),
    gpus: str = typer.Option(None, "--gpus", "--gpu", "-g"),
):
    test_routine(
        config_path=config_path,
        config_cls=FaceGFPGANConfig,
        infer_fn=test_face_gfpgan_fn,
        logdir=logdir,
        checkpoint=checkpoint,
        data_root=data_root,
        output_name=output_name,
        gpus=gpus,
    )


@app.command()
def run_train_face_pix2pixhd(
    config_path: Path = typer.Option(Path("./"), "-c", "--config-path"),
    comment: str = typer.Option(None, "--comment", "-C"),
    logdir: Path = typer.Option(None, "--logdir", "-l"),
    gpus: str = typer.Option(None, "--gpus", "--gpu", "-g"),
    dev_mode: DevMode = typer.Option(DevMode.DISABLED, "--dev-mode", "-m"),
):
    train_routine(
        config_path=config_path,
        config_cls=FacePix2PixHDConfig,
        train_fn=train_face_pix2pixhd_fn,
        comment=comment,
        logdir=logdir,
        gpus=gpus,
        dev_mode=dev_mode,
    )


@app.command()
def run_test_face_pix2pixhd(
    config_path: Path = typer.Option(Path("./"), "-c", "--config-path"),
    logdir: Path = typer.Option(None, "--logdir", "-l"),
    checkpoint: str = typer.Option("best", "-cpt", "--checkpoint"),
    data_root: Path = typer.Option(None, "--data-root", "-d"),
    output_name: str = typer.Option(None, "--output-name", "-o"),
    gpus: str = typer.Option(None, "--gpus", "--gpu", "-g"),
):
    test_routine(
        config_path=config_path,
        config_cls=FacePix2PixHDConfig,
        infer_fn=test_face_pix2pixhd_fn,
        logdir=logdir,
        checkpoint=checkpoint,
        data_root=data_root,
        output_name=output_name,
        gpus=gpus,
    )


@app.command()
def run_train_face_spade(
    config_path: Path = typer.Option(Path("./"), "-c", "--config-path"),
    comment: str = typer.Option(None, "--comment", "-C"),
    logdir: Path = typer.Option(None, "--logdir", "-l"),
    gpus: str = typer.Option(None, "--gpus", "--gpu", "-g"),
    dev_mode: DevMode = typer.Option(DevMode.DISABLED, "--dev-mode", "-m"),
):
    train_routine(
        config_path=config_path,
        config_cls=FaceSpadeConfig,
        train_fn=train_face_spade_fn,
        comment=comment,
        logdir=logdir,
        gpus=gpus,
        dev_mode=dev_mode,
    )


@app.command()
def run_test_face_spade(
    config_path: Path = typer.Option(Path("./"), "-c", "--config-path"),
    logdir: Path = typer.Option(None, "--logdir", "-l"),
    checkpoint: str = typer.Option("best", "-cpt", "--checkpoint"),
    data_root: Path = typer.Option(None, "--data-root", "-d"),
    output_name: str = typer.Option(None, "--output-name", "-o"),
    gpus: str = typer.Option(None, "--gpus", "--gpu", "-g"),
):
    test_routine(
        config_path=config_path,
        config_cls=FaceSpadeConfig,
        infer_fn=test_face_spade_fn,
        logdir=logdir,
        checkpoint=checkpoint,
        data_root=data_root,
        output_name=output_name,
        gpus=gpus,
    )


@app.command()
def run_module_speed_test(
    config_path: Path = typer.Option(Path("./"), "-c", "--config-path"),
    logdir: Path = typer.Option(None, "--logdir", "-l"),
    repeats: int = typer.Option(10000, "--repeats", "-r"),
):
    module_speed_test_routine(
        config_path=config_path,
        config_cls=DeepfakeConfig,
        logdir=logdir,
        repeats=repeats,
    )


@app.command()
def run_jit_speed_test(
    config_path: Path = typer.Option(Path("./"), "-c", "--config-path"),
    logdir: Path = typer.Option(None, "--logdir", "-l"),
    checkpoint: str = typer.Option("best", "--checkpoint", "-cpt"),
    repeats: int = typer.Option(10000, "--repeats", "-r"),
):
    jit_speed_test_routine(
        config_path=config_path,
        config_cls=DeepfakeConfig,
        logdir=logdir,
        checkpoint=checkpoint,
        repeats=repeats,
    )


@app.command()
def run_model_stats(
    config_path: Path = typer.Option(Path("./"), "-c", "--config-path"),
    logdir: Path = typer.Option(None, "--logdir", "-l"),
):
    model_stats_routine(
        config_path=config_path,
        config_cls=DeepfakeConfig,
        logdir=logdir,
    )


if __name__ == "__main__":
    app()
