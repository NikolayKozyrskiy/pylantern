from __future__ import annotations

from typing import Dict, List

from albumentations import BasicTransform, Compose
from albumentations.core.composition import BaseCompose


class SpecificTargetIgnore(Compose):
    """
    Explicitly allow specific_targets to be ignored by augmentation
    """

    def __init__(
        self,
        transforms: list[BasicTransform | BaseCompose],
        specific_targets: List[str] = [],
        additional_targets: Dict[str, str] = {},
    ):
        super().__init__(list(transforms), additional_targets=additional_targets)
        self.specific_targets = specific_targets
        assert isinstance(
            self.specific_targets, list
        ), "specific_targets should be list"

    def __call__(self, **kwargs):
        tmp = {}
        for k in self.specific_targets:
            if k in kwargs:
                tmp[k] = kwargs.pop(k)

        kwargs = Compose.__call__(self, **kwargs)

        kwargs.update(tmp)
        return kwargs
