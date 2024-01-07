from __future__ import annotations

from typing import TYPE_CHECKING, Callable, NamedTuple

import torch
from torch import Tensor

if TYPE_CHECKING:
    from pylantern.tasks.gan.pix2pix.pipelines.general.pipeline import (
        BasePix2PixPipeline,
    )


class PreviewImageConfig(NamedTuple):
    base: Callable
    overlays: list[Callable] = []


def create_preview_images(
    configs: list[PreviewImageConfig], pipeline: "BasePix2PixPipeline"
) -> Tensor:
    images = []
    for conf in configs:
        base = conf.base(pipeline)

        for overlay in conf.overlays:
            base = overlay(base, pipeline)

        images.append(base.cpu())

    return torch.cat(images, dim=-1)
