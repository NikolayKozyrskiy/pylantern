from typing import Union

import torch
from torch import Tensor
from torch.nn import functional as F


def loss_masked_reduce_float(
    loss: Tensor, mask: Tensor, channel_dim: int = -3, reduction_start_dim: int = 1
):
    if mask.ndim != loss.ndim:
        mask = mask.unsqueeze(channel_dim)
    loss = loss.mean(channel_dim, keepdim=True)
    assert loss.ndim == mask.ndim
    loss = loss * mask
    return loss.flatten(reduction_start_dim).sum(dim=-1) / (
        mask.flatten(reduction_start_dim).sum(dim=-1) + 1e-7
    )


def loss_masked_reduce_bool(
    loss: Tensor, mask: Tensor, channel_dim: int = -3, reduction_start_dim: int = 1
):
    assert mask.dtype == torch.bool
    if mask.ndim != loss.ndim:
        mask = mask.unsqueeze(channel_dim)
    loss = loss.mean(channel_dim, keepdim=True)
    assert loss.ndim == mask.ndim
    loss = torch.where(mask, loss, torch.zeros(1, device=loss.device, dtype=loss.dtype))
    return loss.flatten(reduction_start_dim).sum(dim=-1) / (
        mask.flatten(reduction_start_dim).sum(dim=-1) + 1e-7
    )


def loss_masked_reduce(
    loss: Tensor, mask: Tensor, channel_dim: int = -3, reduction_start_dim: int = 1
):
    if mask.dtype == torch.bool:
        return loss_masked_reduce_bool(loss, mask, channel_dim, reduction_start_dim)
    return loss_masked_reduce_float(loss, mask, channel_dim, reduction_start_dim)


def masked_mse(input, target, mask, channel_dim: int = -3):
    mse = F.mse_loss(input, target, reduction="none")
    return loss_masked_reduce(mse, mask, channel_dim=channel_dim)


def masked_l1(
    input: Tensor,
    target: Tensor,
    mask: Tensor,
    channel_dim: int = -3,
    reduction_start_dim: int = 1,
):
    loss = F.l1_loss(input, target, reduction="none")
    return loss_masked_reduce(loss, mask, channel_dim, reduction_start_dim)


def masked_smooth_l1(
    input: Tensor,
    target: Tensor,
    mask: Tensor,
    beta: float = 0.01,
    div: Union[Tensor, None] = None,
    channel_dim: int = -3,
    reduction_start_dim: int = 1,
):
    if div is not None:
        input = input / div
        target = target / div
    loss = F.smooth_l1_loss(input, target, beta=beta, reduction="none")
    return loss_masked_reduce(loss, mask, channel_dim, reduction_start_dim)


def l1_scaled_masked(input, target, scaling, mask):
    loss = F.l1_loss(input, target, reduction="none")
    loss = loss * scaling.expand_as(loss)
    return loss_masked_reduce(loss, mask)
