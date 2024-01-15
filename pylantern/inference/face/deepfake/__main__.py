from pathlib import Path

import typer

from pylantern.common.train_fns import DevMode
from pylantern.inference.face.deepfake.fns import infer_video

app = typer.Typer()


@app.command()
def run_infer_video(
    config_path: Path = typer.Option(Path("./"), "-c", "--config-path"),
    frames_num: int = typer.Option(None, "--frames-num", "-f"),
    dev_mode: DevMode = typer.Option(DevMode.DISABLED, "--dev-mode", "-m"),
):
    if dev_mode == DevMode.SHORT:
        frames_num = 10
    infer_video(config_path=config_path, frames_num=frames_num)


@app.command()
def run_infer_img_folder(
    config_path: Path = typer.Option(Path("./"), "-c", "--config-path"),
    imgs_num: int = typer.Option(None, "--frames-num", "-f"),
    dev_mode: DevMode = typer.Option(DevMode.DISABLED, "--dev-mode", "-m"),
):
    if dev_mode == DevMode.SHORT:
        imgs_num = 3


if __name__ == "__main__":
    app()
