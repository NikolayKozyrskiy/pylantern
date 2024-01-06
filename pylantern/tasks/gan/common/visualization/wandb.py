from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import wandb
from ignite.distributed import one_rank_only
from matches.loop import Loop

from pylantern.common.utils import mkdir
from pylantern.common.utils.img import save_img, tensor_to_image

from .builder import create_preview_images

if TYPE_CHECKING:
    from pylantern.tasks.gan.pix2pix.pipelines.pipeline import GanPix2PixPipeline


def log_images_to_wandb(
    loop: Loop,
    pipeline: "GanPix2PixPipeline",
    prefix: str,
    img_show_num: int = 10,
) -> None:
    with loop.mode(mode="valid"):
        images = create_preview_images(pipeline.config.preview_config, pipeline)

    ep = loop.iterations.current_epoch
    data = {"epochs": ep}
    for name, im in zip(
        pipeline.batch["name"],
        images[:img_show_num],
    ):
        id_ = f"{prefix}/{name}"
        root: Path = mkdir(loop.logdir / f"history/image/{prefix}")

        res_path = root / f"{name}__{ep:03d}.jpg"
        save_img(
            tensor_to_image(im, keepdim=False, val_range=(0.0, 1.0)),
            res_path,
            convert_rgb2bgr=True,
        )
        data[f"images/{id_}"] = wandb.Image(str(res_path.resolve()), caption=f"ep={ep}")

    @one_rank_only()
    def _log():
        wandb.log(
            data,
            commit=False,
        )

    _log()

    return None
