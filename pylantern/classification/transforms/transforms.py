from typing import List, Callable, Union, Tuple

import torchvision.transforms as tr


def train_basic_augs(
    crop_size: Union[int, Tuple[int, int]] = 32, padding: int = 4
) -> List[Callable]:
    return [tr.RandomCrop(crop_size, padding=padding), tr.RandomHorizontalFlip()]
