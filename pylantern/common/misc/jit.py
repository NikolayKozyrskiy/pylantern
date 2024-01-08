from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import Tensor
from torch.nn import Module

from pylantern.common.utils import get_device


@torch.no_grad()
def jit_script_torch_module(
    module: Module,
    output_path: Path,
    example_inputs_shape: Optional[Sequence[int]] = None,
) -> None:
    device = get_device()
    example_inputs = (
        [torch.randn(example_inputs_shape).to(device)]
        if example_inputs_shape is not None
        else None
    )
    module = module.to(device).eval()
    module_jit = torch.jit.script(module, example_inputs=example_inputs)
    torch.jit.save(module_jit, output_path)
    return module_jit


@torch.no_grad()
def jit_trace_torch_module(
    module: Module,
    output_path: Path,
    example_kwarg_inputs: dict[str, Tensor],
) -> None:
    device = get_device()
    for k in example_kwarg_inputs.keys():
        if isinstance(example_kwarg_inputs[k], Tensor) or isinstance(
            example_kwarg_inputs[k], Module
        ):
            example_kwarg_inputs[k] = example_kwarg_inputs[k].to(device)
    module = module.to(device).eval()
    module_jit = torch.jit.trace(module, example_kwarg_inputs=example_kwarg_inputs)
    torch.jit.save(module_jit, output_path)
    return module_jit
