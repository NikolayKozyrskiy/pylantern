from __future__ import annotations

from concurrent.futures import Executor
from pathlib import Path
from typing import TYPE_CHECKING, Callable, NamedTuple

import torch
from torch import Tensor

from pylantern.common.utils import mkdir
from pylantern.common.utils.img import save_img, tensor_to_image
from pylantern.common.visualization.utils import delayed

if TYPE_CHECKING:
    from pylantern.pipeline import BasePipeline


class PreviewImageConfig(NamedTuple):
    base: Callable
    overlays: list[Callable] = []


def create_preview_images(
    configs: list["PreviewImageConfig"], pipeline: "BasePipeline"
) -> Tensor:
    images = []
    for conf in configs:
        base = conf.base(pipeline)

        for overlay in conf.overlays:
            base = overlay(base, pipeline)

        images.append(base.cpu())

    return torch.cat(images, dim=-1)


@delayed
def save_previews(
    pipeline: "BasePipeline",
    io_pool: Executor,
    root: Path,
    name_postfix: str = "",
):
    images_dir = mkdir(root / "previews")
    images = create_preview_images(pipeline.config.preview_config, pipeline)
    images = tensor_to_image(images, keepdim=True, val_range=(0.0, 1.0))

    for name, im_i in zip(pipeline.batch["name"], images):
        io_pool.submit(save_img, im_i, images_dir / f"{name}_{name_postfix}.jpg", True)
