import torch
from torch import Tensor
from torchtyping import TensorType


def correct_labels(
    pred: TensorType["batch_size", "num_classes"], gt: TensorType["batch_size"]
) -> TensorType["batch_size"]:
    return pred.max(1)[1].eq(gt).float()


def psnr(x: Tensor, y: Tensor, max_val: float = 1.0) -> Tensor:
    mse = torch.mean((x - y) ** 2, dim=(1, 2, 3), keepdim=True)
    return 10 * torch.log10(max_val**2 / mse)
