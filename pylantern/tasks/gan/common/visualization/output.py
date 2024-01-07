from __future__ import annotations

import functools
from collections import defaultdict
from concurrent.futures import Executor
from pathlib import Path
from typing import TYPE_CHECKING, Callable, TypeVar

import cv2
import numpy as np
import torch
from typing_extensions import Concatenate, ParamSpec

from pylantern.common.utils import append_json, mkdir
from pylantern.common.utils.img import save_img, tensor_to_image

from .builder import create_preview_images

if TYPE_CHECKING:
    from pylantern.tasks.gan.pix2pix.pipelines.general.pipeline import (
        BasePix2PixPipeline,
    )

Args = ParamSpec("Args")
R = TypeVar("R")


def delayed(
    f: Callable[Concatenate["BasePix2PixPipeline", Executor, str, Args], R]
) -> Callable[[Args], Callable[["BasePix2PixPipeline", Executor, str], R]]:
    @functools.wraps(f)
    def partial(*args: Args.args, **kwargs: Args.kwargs):
        return functools.partial(f, *args, **kwargs)

    # noinspection PyTypeChecker
    return partial


@delayed
def save_previews(
    pipeline: "BasePix2PixPipeline",
    io_pool: Executor,
    root: Path,
    name_postfix: str = "",
):
    images_dir = mkdir(root / "previews")
    images = create_preview_images(pipeline.config.preview_config, pipeline)
    images = tensor_to_image(images, keepdim=True, val_range=(0.0, 1.0))

    for name, im_i in zip(pipeline.batch["name"], images):
        io_pool.submit(save_img, im_i, images_dir / f"{name}_{name_postfix}.jpg", True)


@delayed
def save_output_statistics(
    pipeline: "BasePix2PixPipeline",
    io_pool: Executor,
    root: Path,
    name_postfix: str = "",
):
    output_file_path = root / "statistics.json"
    statistics = defaultdict(list)

    def _merge_fn(d1: dict, d2: dict):
        for k in d1.keys():
            d1[k] += d2[k]

    with torch.no_grad():
        imgs = pipeline.get_predicted_image()
        statistics["mean"].append(
            imgs.mean(dim=(1, 2, 3)).mean().cpu().detach().tolist()
        )
        statistics["std"].append(imgs.std(dim=(1, 2, 3)).mean().cpu().detach().tolist())
        statistics["max_val"].append(
            torch.amax(imgs, dim=(1, 2, 3)).mean().cpu().detach().tolist()
        )
        statistics["min_val"].append(
            torch.amin(imgs, dim=(1, 2, 3)).mean().cpu().detach().tolist()
        )

        statistics["median"].append(
            np.median(imgs.cpu().detach().numpy(), axis=(1, 2, 3)).mean().tolist()
        )
        statistics["q_999"].append(
            np.quantile(imgs.cpu().detach().numpy(), q=0.999, axis=(1, 2, 3))
            .mean()
            .tolist()
        )
        statistics["q_99"].append(
            np.quantile(imgs.cpu().detach().numpy(), q=0.99, axis=(1, 2, 3))
            .mean()
            .tolist()
        )
        statistics["q_98"].append(
            np.quantile(imgs.cpu().detach().numpy(), q=0.98, axis=(1, 2, 3))
            .mean()
            .tolist()
        )

        statistics["q_001"].append(
            np.quantile(imgs.cpu().detach().numpy(), q=0.001, axis=(1, 2, 3))
            .mean()
            .tolist()
        )
        statistics["q_01"].append(
            np.quantile(imgs.cpu().detach().numpy(), q=0.01, axis=(1, 2, 3))
            .mean()
            .tolist()
        )
        statistics["q_02"].append(
            np.quantile(imgs.cpu().detach().numpy(), q=0.02, axis=(1, 2, 3))
            .mean()
            .tolist()
        )

        append_json(statistics, output_file_path, _merge_fn, indent=2)
