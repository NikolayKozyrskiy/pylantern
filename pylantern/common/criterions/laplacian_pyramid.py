import torch.nn as nn
from kornia.geometry.transform import build_laplacian_pyramid
from torch import Tensor


class LaplacianLoss(nn.Module):
    def __init__(self, max_levels: int = 3):
        super(LaplacianLoss, self).__init__()
        self.max_levels = max_levels
        self.l1_loss = nn.L1Loss()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        pyr_input = build_laplacian_pyramid(input=input, max_level=self.max_levels)
        pyr_target = build_laplacian_pyramid(input=target, max_level=self.max_levels)
        return sum(self.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))
