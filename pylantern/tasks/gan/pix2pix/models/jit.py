from pathlib import Path
from typing import Optional

import torch
from torch.nn import Module

from pylantern.common.utils import get_device
from pylantern.config import load_config
from pylantern.tasks.gan.pix2pix.configs import BasePix2PixConfig, FacePix2PixConfig
from pylantern.tasks.gan.pix2pix.models.models import (
    load_face_generator_inference_model,
    load_face_generator_model,
    load_generator_inference_model,
    load_generator_model,
)


@torch.no_grad()
def jit_script_generator_model(
    config_path: "Path",
    checkpoint_path: Optional["Path"] = None,
) -> Module:
    config = load_config(config_path=config_path, desired_class=BasePix2PixConfig)
    checkpoint_path = (
        checkpoint_path if checkpoint_path is not None else config.checkpoint_path
    )
    device = get_device()
    model = load_generator_model(
        config=config,
        checkpoint_path=checkpoint_path,
        is_train=False,
        load_strict=True,
        device=device,
    )
    example_inputs = [
        torch.randn(1, 3, config.image_size[0], config.image_size[1]).to(device)
    ]
    model_jit = torch.jit.script(model, example_inputs=example_inputs)
    torch.jit.save(
        model_jit,
        checkpoint_path.parent / f"{checkpoint_path.stem}_generator_model_jit.pth",
    )
    return model_jit


@torch.no_grad()
def jit_script_generator_inference_model(
    config_path: "Path",
    checkpoint_path: Optional["Path"] = None,
) -> Module:
    config = load_config(config_path=config_path, desired_class=BasePix2PixConfig)
    checkpoint_path = (
        checkpoint_path if checkpoint_path is not None else config.checkpoint_path
    )
    device = get_device()
    model = load_generator_inference_model(
        config=config,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    example_inputs = [
        torch.randn(1, 3, config.image_size[0], config.image_size[1]).to(device)
    ]
    model_jit = torch.jit.script(model, example_inputs=example_inputs)
    torch.jit.save(
        model_jit,
        checkpoint_path.parent
        / f"{checkpoint_path.stem}_generator_inference_model_jit.pth",
    )
    return model_jit


@torch.no_grad()
def jit_script_face_generator_model(
    config_path: "Path",
    checkpoint_path: Optional["Path"] = None,
) -> Module:
    config = load_config(config_path=config_path, desired_class=FacePix2PixConfig)
    checkpoint_path = (
        checkpoint_path if checkpoint_path is not None else config.checkpoint_path
    )
    device = get_device()
    model = load_face_generator_model(
        config=config,
        checkpoint_path=checkpoint_path,
        is_train=False,
        load_strict=True,
        device=device,
    )
    example_inputs = [
        torch.randn(1, 3, config.image_size[0], config.image_size[1]).to(device)
    ]
    model_jit = torch.jit.script(model, example_inputs=example_inputs)
    torch.jit.save(
        model_jit,
        checkpoint_path.parent / f"{checkpoint_path.stem}_face_generator_model_jit.pth",
    )
    return model_jit


@torch.no_grad()
def jit_script_face_generator_inference_model(
    config_path: "Path",
    checkpoint_path: Optional["Path"] = None,
) -> Module:
    config = load_config(config_path=config_path, desired_class=FacePix2PixConfig)
    checkpoint_path = (
        checkpoint_path if checkpoint_path is not None else config.checkpoint_path
    )
    device = get_device()
    model = load_face_generator_inference_model(
        config=config,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    example_inputs = [
        torch.randn(1, 3, config.image_size[0], config.image_size[1]).to(device)
    ]
    model_jit = torch.jit.script(model, example_inputs=example_inputs)
    torch.jit.save(
        model_jit,
        checkpoint_path.parent
        / f"{checkpoint_path.stem}_face_generator_inference_model_jit.pth",
    )
    return model_jit
