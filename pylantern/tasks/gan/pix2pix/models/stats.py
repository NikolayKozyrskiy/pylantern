from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Module

from pylantern.common.misc.model_stats import (
    jit_speed_test,
    model_stats,
    module_speed_test,
)
from pylantern.common.utils import get_device, mkdir
from pylantern.common.utils.module import model_eval
from pylantern.config import load_config
from pylantern.tasks.gan.pix2pix.configs import BasePix2PixConfig, FacePix2PixConfig


def generator_model_module_speed_test(
    config_path: "Path", logdir: "Path", repeats: int = 10000
) -> None:
    config = load_config(config_path=config_path, desired_class=BasePix2PixConfig)
    mkdir(logdir)
    generator_model, inputs = _common_module_speed_test_settings(config=config)
    module_speed_test(
        logdir=logdir,
        model=generator_model,
        inputs=inputs,
        input_sizes=config.image_size,
        repeats=repeats,
        postfix="generator_model",
    )


def generator_model_jit_speed_test(
    config_path: "Path",
    logdir: "Path",
    checkpoint_name: str = "best",
    repeats: int = 10000,
) -> None:
    config = load_config(config_path=config_path, desired_class=BasePix2PixConfig)
    mkdir(logdir)
    inputs = _common_jit_speed_test_settings(config=config)
    jit_speed_test(
        logdir=logdir,
        inputs=inputs,
        input_sizes=config.image_size,
        checkpoint_name=checkpoint_name,
        repeats=repeats,
        postfix="generator_model",
    )


def generator_model_stats(config_path: "Path", logdir: "Path") -> None:
    config = load_config(config_path=config_path, desired_class=BasePix2PixConfig)
    mkdir(logdir)
    generator_model, inputs = _common_model_stats_settings(config=config)
    model_stats(
        logdir=logdir,
        model=generator_model,
        inputs=inputs,
        input_sizes=config.image_size,
        postfix="generator_model",
    )


def generator_inference_model_module_speed_test(
    config_path: "Path", logdir: "Path", repeats: int = 10000
) -> None:
    config = load_config(config_path=config_path, desired_class=BasePix2PixConfig)
    mkdir(logdir)
    generator_model, inputs = _common_module_speed_test_settings(config=config)
    module_speed_test(
        logdir=logdir,
        model=generator_model,
        inputs=inputs,
        input_sizes=config.image_size,
        repeats=repeats,
        postfix="generator_inference_model",
    )


def generator_inference_model_jit_speed_test(
    config_path: "Path",
    logdir: "Path",
    checkpoint_name: str = "best",
    repeats: int = 10000,
) -> None:
    config = load_config(config_path=config_path, desired_class=BasePix2PixConfig)
    mkdir(logdir)
    inputs = _common_jit_speed_test_settings(config=config)
    jit_speed_test(
        logdir=logdir,
        inputs=inputs,
        input_sizes=config.image_size,
        checkpoint_name=checkpoint_name,
        repeats=repeats,
        postfix="generator_inference_model",
    )


def generator_inference_model_stats(config_path: "Path", logdir: "Path") -> None:
    config = load_config(config_path=config_path, desired_class=BasePix2PixConfig)
    mkdir(logdir)
    generator_model, inputs = _common_model_stats_settings(config=config)
    model_stats(
        logdir=logdir,
        model=generator_model,
        inputs=inputs,
        input_sizes=config.image_size,
        postfix="generator_inference_model",
    )


def face_generator_model_module_speed_test(
    config_path: "Path", logdir: "Path", repeats: int = 10000
) -> None:
    config = load_config(config_path=config_path, desired_class=FacePix2PixConfig)
    mkdir(logdir)
    generator_model, inputs = _common_module_speed_test_settings(config=config)
    module_speed_test(
        logdir=logdir,
        model=generator_model,
        inputs=inputs,
        input_sizes=config.image_size,
        repeats=repeats,
        postfix="face_generator_model",
    )


def face_generator_model_jit_speed_test(
    config_path: "Path",
    logdir: "Path",
    checkpoint_name: str = "best",
    repeats: int = 10000,
) -> None:
    config = load_config(config_path=config_path, desired_class=FacePix2PixConfig)
    mkdir(logdir)
    inputs = _common_jit_speed_test_settings(config=config)
    jit_speed_test(
        logdir=logdir,
        inputs=inputs,
        input_sizes=config.image_size,
        checkpoint_name=checkpoint_name,
        repeats=repeats,
        postfix="face_generator_model",
    )


def face_generator_model_stats(config_path: "Path", logdir: "Path") -> None:
    config = load_config(config_path=config_path, desired_class=FacePix2PixConfig)
    mkdir(logdir)
    generator_model, inputs = _common_model_stats_settings(config=config)
    model_stats(
        logdir=logdir,
        model=generator_model,
        inputs=inputs,
        input_sizes=config.image_size,
        postfix="face_generator_model",
    )


def face_generator_inference_model_module_speed_test(
    config_path: "Path", logdir: "Path", repeats: int = 10000
) -> None:
    config = load_config(config_path=config_path, desired_class=FacePix2PixConfig)
    mkdir(logdir)
    generator_model, inputs = _common_module_speed_test_settings(config=config)
    module_speed_test(
        logdir=logdir,
        model=generator_model,
        inputs=inputs,
        input_sizes=config.image_size,
        repeats=repeats,
        postfix="face_generator_inference_model",
    )


def face_generator_inference_model_jit_speed_test(
    config_path: "Path",
    logdir: "Path",
    checkpoint_name: str = "best",
    repeats: int = 10000,
) -> None:
    config = load_config(config_path=config_path, desired_class=FacePix2PixConfig)
    mkdir(logdir)
    inputs = _common_jit_speed_test_settings(config=config)
    jit_speed_test(
        logdir=logdir,
        inputs=inputs,
        input_sizes=config.image_size,
        checkpoint_name=checkpoint_name,
        repeats=repeats,
        postfix="face_generator_inference_model",
    )


def face_generator_inference_model_stats(config_path: "Path", logdir: "Path") -> None:
    config = load_config(config_path=config_path, desired_class=FacePix2PixConfig)
    mkdir(logdir)
    generator_model, inputs = _common_model_stats_settings(config=config)
    model_stats(
        logdir=logdir,
        model=generator_model,
        inputs=inputs,
        input_sizes=config.image_size,
        postfix="face_generator_inference_model",
    )


def _common_module_speed_test_settings(
    config: "BasePix2PixConfig",
) -> Tuple[Module, Tensor]:
    device = get_device()
    generator_model = model_eval(
        model=config.generator_model(),
        requires_grad=False,
    ).to(device)
    inputs = torch.randn(1, 3, config.image_size[0], config.image_size[1]).to(device)
    return generator_model, inputs


def _common_jit_speed_test_settings(config: "BasePix2PixConfig") -> Tensor:
    return torch.randn(1, 3, config.image_size[0], config.image_size[1]).to(
        get_device()
    )


def _common_model_stats_settings(
    config: "BasePix2PixConfig",
) -> Tuple[Module, Tensor]:
    device = get_device()
    generator_model = model_eval(
        model=config.generator_model(),
        requires_grad=False,
    ).to(device)
    inputs = torch.randn(1, 3, config.image_size[0], config.image_size[1]).to(device)
    return generator_model, inputs
