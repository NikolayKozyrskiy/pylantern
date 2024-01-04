from typing import Optional, Tuple, Union

import torch
from torch import Tensor


def standardize_tensor(
    tensor: Tensor,
    mean: Tensor,
    std: Tensor,
) -> Tensor:
    return (tensor - mean.expand_as(tensor)) / std.expand_as(tensor)


def destandardize_tensor(
    tensor: Tensor,
    mean: Tensor,
    std: Tensor,
) -> Tensor:
    return tensor * std.expand_as(tensor) + mean.expand_as(tensor)


def normalize_tensor(
    tensor: Tensor,
    scale: Optional[Tensor] = None,
    mean: Optional[Tensor] = None,
    std: Optional[Tensor] = None,
) -> Tensor:
    if scale is not None:
        tensor = tensor / scale
    if mean is not None:
        tensor = tensor - mean.expand_as(tensor)
    if std is not None:
        tensor = tensor / std.expand_as(tensor)
    return tensor


def denormalize_tensor(
    tensor: Tensor,
    scale: Optional[Tensor] = None,
    mean: Optional[Tensor] = None,
    std: Optional[Tensor] = None,
    clip_range: Union[Tuple[float, float], float, None] = None,
) -> Tensor:
    if std is not None:
        tensor = tensor * std.expand_as(tensor)
    if mean is not None:
        tensor = tensor + mean.expand_as(tensor)
    if scale is not None:
        tensor = tensor * scale
    if clip_range is not None:
        clip_range = [0.0, clip_range] if isinstance(clip_range, float) else clip_range
        tensor = torch.clip(tensor, clip_range[0], clip_range[1])
    return tensor
