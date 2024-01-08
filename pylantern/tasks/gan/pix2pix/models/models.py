from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Union

import torch
from torch.nn import Module

from pylantern.common.utils.module import (
    model_eval,
    remove_module_from_state_dict,
    set_requires_grad,
    to_device,
)
from pylantern.tasks.gan.pix2pix.models.generators.face_generator import (
    FaceGeneratorInferenceModel,
    FaceGeneratorModel,
)
from pylantern.tasks.gan.pix2pix.models.generators.generator import (
    GeneratorInferenceModel,
    GeneratorModel,
)

if TYPE_CHECKING:
    from pylantern.tasks.gan.pix2pix.configs import BasePix2PixConfig, FacePix2PixConfig


def get_generator_model(
    generator: "Module",
    mean: Sequence[float],
    std: Sequence[float],
    input_range: Sequence[int],
    is_train: bool = True,
    device: Union[str, torch.device, None] = None,
) -> "GeneratorModel":
    model = GeneratorModel(
        generator_model=generator, mean=mean, std=std, input_range=input_range
    )
    model = model.train() if is_train else model.eval()
    model = set_requires_grad(module=model, value=is_train)
    model = to_device(model, device=device)
    return model


def load_generator_model(
    config: "BasePix2PixConfig",
    checkpoint_path: Optional[Path] = None,
    is_train: bool = True,
    load_strict: bool = True,
    device: Union[str, torch.device, None] = None,
) -> Union["GeneratorModel", "GeneratorInferenceModel"]:
    model = config.generator_model()
    checkpoint_path = (
        checkpoint_path if checkpoint_path is not None else config.checkpoint_path
    )
    state_dict_loaded = torch.load(checkpoint_path, map_location="cpu")
    if "generator_model" in state_dict_loaded.keys():
        state_dict_loaded = state_dict_loaded["generator_model"]
    state_dict_loaded = remove_module_from_state_dict(state_dict_loaded)

    state_dict = OrderedDict()
    for k, v in state_dict_loaded.items():
        if not k.startswith("generator_model."):
            state_dict[f"generator_model.{k}"] = v
        else:
            state_dict[k] = v

    model.load_state_dict(state_dict, strict=load_strict)
    model = model.train() if is_train else model.eval()
    model = set_requires_grad(module=model, value=is_train)
    model = to_device(model, device=device)
    return model


def get_generator_inference_model(
    generator: "Module",
    mean: Sequence[float],
    std: Sequence[float],
    state_dict: Optional[dict] = None,
    device: Union[str, torch.device, None] = None,
) -> "GeneratorInferenceModel":
    model = model_eval(
        model=GeneratorInferenceModel(generator_model=generator, mean=mean, std=std),
        state_dict=remove_module_from_state_dict(state_dict=state_dict),
        requires_grad=False,
    )
    model = to_device(model, device=device)
    return model


def load_generator_inference_model(
    config: "BasePix2PixConfig",
    checkpoint_path: Optional[Path] = None,
    device: Union[str, torch.device, None] = None,
) -> "GeneratorInferenceModel":
    return load_generator_model(
        config=config,
        checkpoint_path=checkpoint_path,
        is_train=False,
        load_strict=False,
        device=device,
    )


def get_face_generator_model(
    generator: "Module",
    mean: Sequence[float],
    std: Sequence[float],
    input_range: Sequence[int],
    is_train: bool = True,
    device: Union[str, torch.device, None] = None,
) -> "FaceGeneratorModel":
    model = FaceGeneratorModel(
        generator_model=generator, mean=mean, std=std, input_range=input_range
    )
    model = model.train() if is_train else model.eval()
    model = set_requires_grad(module=model, value=is_train)
    model = to_device(model, device=device)
    return model


def load_face_generator_model(
    config: "FacePix2PixConfig",
    checkpoint_path: Optional[Path] = None,
    is_train: bool = True,
    load_strict: bool = True,
    device: Union[str, torch.device, None] = None,
) -> Union["FaceGeneratorModel", "FaceGeneratorInferenceModel"]:
    model = config.generator_model()
    checkpoint_path = (
        checkpoint_path if checkpoint_path is not None else config.checkpoint_path
    )
    state_dict_loaded = torch.load(checkpoint_path, map_location="cpu")
    if "generator_model" in state_dict_loaded.keys():
        state_dict_loaded = state_dict_loaded["generator_model"]
    state_dict_loaded = remove_module_from_state_dict(state_dict_loaded)

    state_dict = OrderedDict()
    for k, v in state_dict_loaded.items():
        if not k.startswith("generator_model."):
            state_dict[f"generator_model.{k}"] = v
        else:
            state_dict[k] = v

    model.load_state_dict(state_dict, strict=load_strict)
    model = model.train() if is_train else model.eval()
    model = set_requires_grad(module=model, value=is_train)
    model = to_device(model, device=device)
    return model


def get_face_generator_inference_model(
    generator: "Module",
    mean: Sequence[float],
    std: Sequence[float],
    state_dict: Optional[dict] = None,
    device: Union[str, torch.device, None] = None,
) -> "FaceGeneratorInferenceModel":
    model = model_eval(
        model=FaceGeneratorInferenceModel(
            generator_model=generator, mean=mean, std=std
        ),
        state_dict=remove_module_from_state_dict(state_dict=state_dict),
        requires_grad=False,
    )
    model = to_device(model, device=device)
    return model


def load_face_generator_inference_model(
    config: "FacePix2PixConfig",
    checkpoint_path: Optional[Path] = None,
    device: Union[str, torch.device, None] = None,
) -> "FaceGeneratorInferenceModel":
    return load_face_generator_model(
        config=config,
        checkpoint_path=checkpoint_path,
        is_train=False,
        load_strict=False,
        device=device,
    )
