from typing import List

import torchvision.transforms as tr


def wrap_transforms(transforms: List) -> tr.Compose:
    return tr.Compose(transforms=transforms + to_tensor())


def to_tensor() -> List:
    return [tr.ToTensor()]


def to_tensor_normalized() -> List:
    return [
        tr.ToTensor(),
        tr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
