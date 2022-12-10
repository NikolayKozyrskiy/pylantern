from __future__ import annotations

import functools
from concurrent.futures import Executor
from pathlib import Path
from typing import Callable, TypeVar, NamedTuple, List

import imageio
import numpy as np
import torch
from typing_extensions import Concatenate, ParamSpec
from matches.loop import Loop
import wandb

from .utils import tensor_to_image
from ..pipeline import Pipeline

Args = ParamSpec("Args")
R = TypeVar("R")


def delayed(
    f: Callable[Concatenate[Pipeline, Executor, Path, Args], R]
) -> Callable[[Args], Callable[[Pipeline, Executor, Path], R]]:
    @functools.wraps(f)
    def partial(*args: Args.args, **kwargs: Args.kwargs):
        return functools.partial(f, *args, **kwargs)

    # noinspection PyTypeChecker
    return partial


@delayed
def save_previews(
    pipeline: Pipeline,
    io_pool: Executor,
    root: Path,
    name_postfix: str = "",
):
    images_dir = root / "preview"
    images_dir.mkdir(parents=True, exist_ok=True)
    images = create_preview_images(pipeline.config.preview_image_fns, pipeline)
    images = tensor_to_image(images, keepdim=True)

    for name, im_i in zip(pipeline.batch["name"], images):
        io_pool.submit(imageio.imwrite, images_dir / f"{name}_{name_postfix}.jpg", im_i)


def create_preview_images(
    preview_image_fns: List[Callable], pipeline: Pipeline
) -> torch.Tensor:
    images = []
    for image_fn in preview_image_fns:
        images.append(image_fn(pipeline).cpu())

    return torch.cat(images, dim=-1)


def log_images_to_wandb(loop: Loop, pipeline: Pipeline, prefix: str):
    with loop.mode():
        images = create_preview_images(pipeline.config.preview_image_fns, pipeline)

    ep = loop.iterations.current_epoch
    data = {"epochs": ep}
    for name, im in zip(
        pipeline.batch.name,
        images[:10],
    ):
        id_ = f"{prefix}/{name}"
        root = loop.logdir / f"history/image/{prefix}"
        root.mkdir(exist_ok=True, parents=True)

        res_path = root / f"{name}__{ep:03d}.jpg"
        imageio.v3.imwrite(res_path, tensor_to_image(im, keepdim=False))
        data[f"images/{id_}"] = wandb.Image(str(res_path.resolve()), caption=f"ep={ep}")

    wandb.log(
        data,
        commit=False,
    )
