from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from albumentations import BasicTransform


class FilterMask(BasicTransform):
    def __init__(
        self,
        classes_to_keep: Optional[Sequence[int]] = None,
        classes_to_remove: Optional[Sequence[int]] = None,
        mask_name: str = "mask",
    ):
        super().__init__(p=1.0)
        self.classes_to_remove = (
            None if classes_to_remove is None else np.array(classes_to_remove)
        )
        self.classes_to_keep = (
            None if classes_to_keep is None else np.array(classes_to_keep)
        )
        self.mask_name = mask_name

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return "classes_to_keep", "classes_to_remove", "mask_name"

    @property
    def targets(self):
        return {self.mask_name: self.filter_mask}

    def filter_mask(self, mask, **kwargs):
        result = np.ones_like(mask)

        if self.classes_to_keep is not None:
            result &= (mask[..., None] == self.classes_to_keep).any(axis=-1)
        if self.classes_to_remove is not None:
            result &= (mask[..., None] != self.classes_to_remove).all(axis=-1)
        return result


class SubtractMasks(BasicTransform):
    def __init__(
        self,
        primary_mask: str = "mask",
        secondary_masks: Optional[Sequence[str]] = None,
    ):
        super().__init__(p=1.0)
        self.primary_mask = primary_mask
        self.secondary_masks = secondary_masks

    def __call__(self, *args, force_apply: bool = False, **kwargs) -> Dict[str, Any]:
        if self.secondary_masks is not None:
            for secondary_mask in self.secondary_masks:
                kwargs[self.primary_mask] -= kwargs[secondary_mask]
        return kwargs

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return "primary_mask", "secondary_masks"
