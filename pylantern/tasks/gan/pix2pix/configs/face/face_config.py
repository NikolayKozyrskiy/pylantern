from typing import TYPE_CHECKING, Optional

from matches.shortcuts.optimizer import LRSchedulerWrapper, SchedulerScopeType
from torch.nn import Module
from torch.optim import Adam, Optimizer

from .. import BasePix2PixConfig, GFPGANConfig, Pix2PixHDConfig, SpadeConfig

if TYPE_CHECKING:
    from pylantern.model_zoo.gfpgan import FacialComponentDiscriminator


class FacePix2PixConfig(BasePix2PixConfig):
    eye_bbox_enlarge_ratio: float = 1.5
    eye_roi_relative_size: float = 0.15625
    mouth_roi_relative_size: float = 0.234375

    def discriminator_left_eye_model(
        self, **kwargs
    ) -> Optional["FacialComponentDiscriminator"]:
        return None

    def discriminator_right_eye_model(
        self, **kwargs
    ) -> Optional["FacialComponentDiscriminator"]:
        return None

    def discriminator_mouth_model(
        self, **kwargs
    ) -> Optional["FacialComponentDiscriminator"]:
        return None

    def optimizer_discriminator_left_eye(self, model: "Module") -> "Optimizer":
        return Adam(model.parameters(), lr=self.lr)

    def optimizer_discriminator_right_eye(self, model: "Module") -> "Optimizer":
        return Adam(model.parameters(), lr=self.lr)

    def optimizer_discriminator_mouth(self, model: "Module") -> "Optimizer":
        return Adam(model.parameters(), lr=self.lr)

    def scheduler_discriminator_left_eye(
        self, optimizer: "Optimizer"
    ) -> "LRSchedulerWrapper":
        return LRSchedulerWrapper(None, scope_type=SchedulerScopeType.BATCH)

    def scheduler_discriminator_right_eye(
        self, optimizer: "Optimizer"
    ) -> "LRSchedulerWrapper":
        return LRSchedulerWrapper(None, scope_type=SchedulerScopeType.BATCH)

    def scheduler_discriminator_mouth(
        self, optimizer: "Optimizer"
    ) -> "LRSchedulerWrapper":
        return LRSchedulerWrapper(None, scope_type=SchedulerScopeType.BATCH)


class FaceGFPGANConfig(GFPGANConfig, FacePix2PixConfig):
    pass


class FacePix2PixHDConfig(Pix2PixHDConfig, FacePix2PixConfig):
    pass


class FaceSpadeConfig(SpadeConfig, FacePix2PixConfig):
    pass
