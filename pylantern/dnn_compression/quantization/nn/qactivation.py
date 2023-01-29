from typing import Any, Dict, Optional

import torch
from torch import Tensor
import torch.nn as nn


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: Tensor, args_tensor: Any):
        n_bits = args_tensor[0].item()
        is_symmetric = True if input.min() < 0 else False
        if is_symmetric:
            bound = 2 ** (n_bits - 1) - 1
        else:
            bound = 2**n_bits - 1

        scales = (
            torch.max(torch.max(torch.max(torch.abs(input), 3)[0], 2)[0], 1)[0] / bound
        )
        scales = torch.reshape(scales, (-1, 1, 1, 1))

        if is_symmetric:
            output = torch.clamp(torch.round(input / scales), -bound, bound) * scales
        else:
            output = torch.clamp(torch.round(input / scales), 0, bound) * scales

        ctx.save_for_backward(input, output, args_tensor)
        return output

    @staticmethod
    def backward(ctx: Any, grad_outputs: Any) -> Any:
        input, output, args_tensor = ctx.saved_tensors
        n_bits = args_tensor[0].item()
        is_symmetric = True if input.min() < 0 else False
        if is_symmetric:
            bound = 2 ** (n_bits - 1) - 1
        else:
            bound = 2**n_bits - 1

        scales = (
            torch.max(torch.max(torch.max(torch.abs(input), 3)[0], 2)[0], 1)[0] / bound
        )
        scales = torch.reshape(scales, (-1, 1, 1, 1))

        if is_symmetric:
            zeros = -(torch.abs(input / scales - bound).sign() - 1) / 2
        else:
            zeros = -((input / scales - bound).sign() - 1) / 2
        grad_outputs *= zeros
        return grad_outputs, None, None


class QActivation(nn.Module):
    def __init__(
        self,
        num_bits: int,
        name: Optional[str] = None,
        is_active: bool = True,
        device: str = "cuda:0",
    ):
        super(QActivation, self).__init__()
        self.num_bits = num_bits
        self.name = name
        self.is_active = is_active
        self.device = device
        self.args_tensor = self.fill_args_tensor().to(self.device)

        if self.num_bits >= 32:
            self.quantization = None
        else:
            self.quantization = STEFunction().apply

    def forward(self, x):
        if self.quantization is None or not self.is_active:
            return x
        else:
            if x.device != self.device:
                self.device = x.device
                self.args_tensor = self.args_tensor.to(self.device)
            return self.quantization(x, self.args_tensor)

    def fill_args_tensor(self):
        args_tensor = Tensor([self.num_bits])
        return args_tensor

    def __str__(self):
        return (
            f"QActivation:: name: {self.name}, "
            f"num_bits: {self.num_bits}, "
            f"is_active: {self.is_active}, "
            f"device: {self.device}"
        )

    def __repr__(self):
        return (
            f"QActivation:: name: {self.name}, "
            f"num_bits: {self.num_bits}, "
            f"is_active: {self.is_active}, "
            f"device: {self.device}"
        )
