from typing import Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SobelOperatorLoss(nn.Module):
    def __init__(
        self, device: Union[str, torch.device], reduction: str = "mean"
    ) -> None:
        super().__init__()
        self.reduction = reduction
        Gx = torch.tensor(
            [[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]]
        ).float()
        Gy = torch.tensor(
            [[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]]
        ).float()
        self.filter = (
            torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0).unsqueeze(1).to(device)
        )

    def forward(self, predict: Tensor, target: Tensor) -> Tensor:
        predict_filtered = self.__apply_filter(predict)
        target_filtered = self.__apply_filter(target)
        return F.smooth_l1_loss(
            predict_filtered, target_filtered, beta=0.01, reduction=self.reduction
        )

    def __apply_filter(self, x: Tensor) -> Tensor:
        out = F.conv2d(x, self.filter, stride=1, padding=0)
        out = torch.mul(out, out)
        out = torch.sum(out, dim=1, keepdim=True)
        out = torch.sqrt(out)
        return out
