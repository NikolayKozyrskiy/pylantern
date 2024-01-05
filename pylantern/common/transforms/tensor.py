import albumentations as albu
import numpy as np
import torch


class ToFloatTensor(albu.BasicTransform):
    def __init__(self, transpose_mask=False, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.transpose_mask = transpose_mask

    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
        }

    def apply(self, img, **params):
        if len(img.shape) not in [2, 3]:
            raise ValueError("Albumentations only supports images in HW or HWC format")

        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)

        return torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32)

    def apply_to_mask(self, mask, **params):
        if self.transpose_mask and mask.ndim == 3:
            mask = mask.transpose(2, 0, 1)
        return torch.tensor(mask, dtype=torch.float32)

    def apply_to_masks(self, masks, **params):
        return [self.apply_to_mask(mask, **params) for mask in masks]

    def get_transform_init_args_names(self):
        return ("transpose_mask",)

    def get_params_dependent_on_targets(self, params):
        return {}
