from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch
from torch import Tensor, nn

from ..config import DeepfakeConfig, GFPGANConfig, SpadeConfig, load_config
from ..utils.model import load_avaturn_face_swapper, set_requires_grad
from ..utils.utils import get_device


@torch.no_grad()
def jit_script_avaturn_swapper_model(
    config: DeepfakeConfig,
    checkpoint_path: Optional[Path] = None,
) -> nn.Module:
    device = get_device()
    face_swapper = load_avaturn_face_swapper(
        config=config, checkpoint_path=checkpoint_path, device=device
    )
    face_swapper.eval()
    example_inputs = [
        torch.randn(1, 3, config.image_size[0], config.image_size[1]).to(device)
    ]
    face_swapper_jit = torch.jit.script(face_swapper, example_inputs=example_inputs)
    torch.jit.save(
        face_swapper_jit, checkpoint_path.parent / f"{checkpoint_path.stem}_jit.pth"
    )
    return face_swapper_jit


@torch.no_grad()
def jit_script_torch_module(
    module: nn.Module,
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
    module: nn.Module,
    output_path: Path,
    example_kwarg_inputs: dict[str, Tensor],
) -> None:
    device = get_device()
    for k in example_kwarg_inputs.keys():
        if isinstance(example_kwarg_inputs[k], Tensor) or isinstance(
            example_kwarg_inputs[k], nn.Module
        ):
            example_kwarg_inputs[k] = example_kwarg_inputs[k].to(device)
    module = module.to(device).eval()
    module_jit = torch.jit.trace(module, example_kwarg_inputs=example_kwarg_inputs)
    torch.jit.save(module_jit, output_path)
    return module_jit
