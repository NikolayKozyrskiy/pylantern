from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Union

import insightface as infa
import torch
from torch.nn import Module

from pylantern.common.utils.module import (
    model_eval,
    remove_module_from_state_dict,
    set_requires_grad,
    to_device,
)
from pylantern.tasks.gan.pix2pix.models import get_face_generator_inference_model

if TYPE_CHECKING:
    from insightface.model_zoo.inswapper import INSwapper

    from pylantern.inference.face.deepfake.df_config import DeepFakeInferenceConfig
    from pylantern.tasks.gan.pix2pix.models import GeneratorInferenceModel


def load_inswapper128_onnx(
    model_path: Optional["Path"] = None,
) -> "INSwapper":
    model_path = (
        Path("_d") / "infa_checkpoints" / "inswapper_128.onnx"
        if model_path is None
        else model_path
    )
    return infa.model_zoo.get_model(name=str(model_path), download=False)


def load_generator_inference_model(
    generator_model: "Module",
    mean: Sequence[float],
    std: Sequence[float],
    checkpoint_path: "Path",
    device: Union[str, torch.device, None] = None,
) -> "GeneratorInferenceModel":
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

    model = get_face_generator_inference_model(
        generator=generator_model,
        mean=mean,
        std=std,
        state_dict=None,
        device=device,
    )
    model.load_state_dict(state_dict, strict=False)

    return model
