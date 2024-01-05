from __future__ import annotations

from albumentations import BasicTransform, OneOf
from albumentations.core.composition import BaseCompose


class KeypointGuard(OneOf):
    """
    Explicitly allow keypoints to be ignored by augmentation
    """

    def __init__(
        self, transforms: list[BasicTransform | BaseCompose], specific_targets=()
    ):
        super().__init__(list(transforms))
        self.specific_targets = specific_targets
        self._additional_targets = {}

    def __call__(self, **kwargs):
        if len(self.specific_targets):
            kw = self.specific_targets
        else:
            kw = [
                target
                for target, type in self._additional_targets.items()
                if type == "keypoints"
            ]
            kw.append("keypoints")

        tmp = {}
        for k in kw:
            if k in kwargs:
                tmp[k] = kwargs.pop(k)

        kwargs = OneOf.__call__(self, **kwargs)

        kwargs.update(tmp)
        return kwargs

    def add_targets(self, additional_targets):
        """Add targets to transform them the same way as one of existing targets
        ex: {'target_image': 'image'}
        ex: {'obj1_mask': 'mask', 'obj2_mask': 'mask'}
        by the way you must have at least one object with key 'image'

        Args:
            additional_targets (dict): keys - new target name, values - old target name. ex: {'image2': 'image'}
        """
        self._additional_targets.update(additional_targets)
        OneOf.add_targets(self, additional_targets)
