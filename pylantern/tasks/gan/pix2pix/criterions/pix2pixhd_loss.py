from typing import List, Union

from torch import Tensor


def compute_pix2pix_hd_mse(
    input: Union[List[Tensor], List[List[Tensor]]],
    target: float,
    reduction: str = "mean",
) -> Union[List[Tensor], Tensor]:
    if isinstance(input[0], list):
        loss = 0.0
        for input_i in input:
            pred = input_i[-1]
            _cur_loss = (pred - target) ** 2
            if reduction == "mean":
                loss = loss + _cur_loss.mean()
            elif reduction == "sum":
                loss = loss + _cur_loss.sum()
            else:
                loss = loss + _cur_loss
        return loss
    else:
        if reduction == "mean":
            return ((input[-1] - target) ** 2).mean()
        elif reduction == "sum":
            return ((input[-1] - target) ** 2).sum()
        else:
            return (input[-1] - target) ** 2
