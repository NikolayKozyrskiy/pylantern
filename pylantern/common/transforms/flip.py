import random
from typing import Any, Dict, List, Tuple

import albumentations as albu


class HorizontalFlipOrderAware(albu.HorizontalFlip):
    """Horizontal Flip with preserving the order of given elements.

    Args:
        swap_pairs (List[Tuple[str, str]]): Elements to swap after. Default: [].

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        swap_pairs: List[Tuple[str, str]] = [],
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.swap_pairs = swap_pairs

    def __call__(self, *args, force_apply: bool = False, **kwargs) -> Dict[str, Any]:
        if args:
            raise KeyError(
                "You have to pass data to augmentations as named arguments, for example: aug(image=image)"
            )
        if self.replay_mode:
            if self.applied_in_replay:
                return self._swap(self.apply_with_params(self.params, **kwargs))

            return kwargs

        if (random.random() < self.p) or self.always_apply or force_apply:
            kwargs = super().__call__(*args, force_apply=True, **kwargs)
            # print(kwargs)
            return self._swap(kwargs)
        else:
            return kwargs

    def _swap(self, kwargs) -> Dict[str, Any]:
        for pair in self.swap_pairs:
            key_1, key_2 = pair
            if key_1 in kwargs.keys() and key_2 in kwargs.keys():
                kwargs[key_1], kwargs[key_2] = kwargs[key_2], kwargs[key_1]
        return kwargs
