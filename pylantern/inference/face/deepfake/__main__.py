from pathlib import Path

import typer

from pylantern.common.train_fns import DevMode
from pylantern.inference.face.deepfake.fns import infer_image, infer_video

app = typer.Typer()


@app.command()
def run_infer_video(
    config_path: Path = typer.Option(Path("./"), "-c", "--config-path"),
    frames_num: int = typer.Option(None, "--frames-num", "-n"),
    dev_mode: DevMode = typer.Option(DevMode.DISABLED, "--dev-mode", "-m"),
):
    if dev_mode == DevMode.SHORT:
        frames_num = 10
    infer_video(config_path=config_path, frames_num=frames_num)


@app.command()
def run_infer_image(
    config_path: Path = typer.Option(Path("./"), "-c", "--config-path"),
    images_num: int = typer.Option(None, "--images-num", "-n"),
    dev_mode: DevMode = typer.Option(DevMode.DISABLED, "--dev-mode", "-m"),
):
    if dev_mode == DevMode.SHORT:
        images_num = 3
    infer_image(config_path=config_path, images_num=images_num)


if __name__ == "__main__":
    app()
