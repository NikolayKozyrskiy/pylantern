from typing import List, Callable, Union, Tuple

import torchvision.transforms as tr

# import albumentations as al


def train_basic_augs(
    crop_size: Union[int, Tuple[int, int]] = 32, padding: int = 4
) -> List[Callable]:
    return [tr.RandomCrop(crop_size, padding=padding), tr.RandomHorizontalFlip()]


# def train_basic_augs(
#     crop_size: Union[int, Tuple[int, int]] = 32
# ) -> List[Callable]:
#     return [al.RandomCrop(height=crop_size[0], width=crop_size[1]), al.HorizontalFlip()]
