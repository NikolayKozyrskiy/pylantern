from copy import deepcopy
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import torch
from torch.nn import Module


def to_device(
    obj: Union[Module, torch.Tensor, None], device: Union[str, torch.device, None]
) -> Union[Module, torch.Tensor, None]:
    if obj is not None and device is not None:
        obj = obj.to(device)
    return obj


def remove_module_from_state_dict(state_dict: Optional[dict]) -> Optional[dict]:
    if state_dict is None:
        return None
    for k, v in deepcopy(state_dict).items():
        if k.startswith("module."):
            state_dict[k[7:]] = v
            state_dict.pop(k)
    return state_dict


def set_requires_grad(module: Optional[Module], value: bool = False) -> Module:
    if module is not None:
        for p in module.parameters():
            p.requires_grad = value
    return module


def model_eval(
    model: Module, state_dict: Optional[dict] = None, requires_grad: bool = False
) -> Module:
    if state_dict is not None:
        model.load_state_dict(state_dict)
    model.eval()
    if not requires_grad:
        model = set_requires_grad(model, value=False)
    return model


def convert_torch2onnx_tracing(
    torch_model: Module,
    input_shape: Sequence[int],
    onnx_model_path: Path,
    device: Union[str, torch.device],
    input_names: Sequence[str] = ["input"],
    output_names: Sequence[str] = ["output"],
) -> None:
    torch_model = torch_model.to(device)
    torch_model.eval()
    for m in torch_model.modules():
        m.eval()
    torch_input = torch.randn(*input_shape).to(device)
    torch.onnx.export(
        model=torch_model,  # model being run
        args=torch_input,  # model input (or a tuple for multiple inputs)
        f=onnx_model_path,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=16,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=input_names,  # the model's input names
        output_names=output_names,  # the model's output names
        dynamic_axes=None,
    )
    return None


def convert_torch2onnx_dynamo(
    torch_model: Module, input_shape: Sequence[int], onnx_model_path: Path
) -> None:
    torch_model.eval()
    torch_input = torch.randn(*input_shape)
    try:
        onnx_model = torch.onnx.dynamo_export(torch_model, torch_input)
        onnx_model.save(onnx_model_path)
    except:
        print(
            f"This func {convert_torch2onnx_dynamo.__name__} works only for Pytorch >= 2.1"
        )
    return None


def deep_network_interpolation(
    net_a_path: Union[Path, str],
    net_b_path: Union[Path, str],
    dni_weight: Tuple[int, int],
    key: str = "params",
):
    """Deep network interpolation.

    ``Paper: Deep Network Interpolation for Continuous Imagery Effect Transition``
    """
    net_a = torch.load(net_a_path, map_location=torch.device("cpu"))
    net_b = torch.load(net_b_path, map_location=torch.device("cpu"))
    for k, v_a in net_a[key].items():
        net_a[key][k] = dni_weight[0] * v_a + dni_weight[1] * net_b[key][k]
    return net_a
