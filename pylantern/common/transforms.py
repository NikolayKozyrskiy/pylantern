from typing import List

import torchvision.transforms as tr
# import albumentations as al
# from albumentations.pytorch import ToTensorV2


def wrap_transforms(transforms: List) -> tr.Compose:
    return tr.Compose(transforms=transforms + to_tensor_normalized())


def to_tensor() -> List:
    return [tr.ToTensor()]


def to_tensor_normalized() -> List:
    return [
        tr.ToTensor(),
        tr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]


# def wrap_transforms(transforms: List[al.BasicTransform]):
#     return al.Compose(transforms=transforms + to_tensor_normalized_3c())


# def to_tensor_normalized_3c() -> List[al.BasicTransform]:
#     return [
#         al.Normalize(
#             mean=(0.485, 0.456, 0.406),
#             std=(0.229, 0.224, 0.225),
#             max_pixel_value=255.0,
#         ),
#         ToTensorV2(),
#     ]


# def to_tensor_normalized_1c() -> List[al.BasicTransform]:
#     return [
#         al.Normalize(
#             mean=(0.5),
#             std=(0.5),
#             max_pixel_value=255.0,
#         ),
#         ToTensorV2(),
#     ]


# def to_tensor() -> List[al.BasicTransform]:
#     return [ToTensorV2()]
