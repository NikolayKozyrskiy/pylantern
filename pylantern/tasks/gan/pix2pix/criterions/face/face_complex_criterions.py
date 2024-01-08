from typing import Dict

from basicsr.losses.gan_loss import GANLoss
from torch.nn import Module

from pylantern.model_zoo.models import (
    arcface_iresnet18,
    arcface_iresnet34,
    arcface_iresnet50,
    arcface_iresnet100,
    face_clip_vitb16,
    gfpgan_arcface_resnet18,
)


def face_clip(*args, **kwargs) -> Module:
    return face_clip_vitb16(
        root_dir=kwargs.get("face_clip_weights_dir", "_d/clip"),
        requires_grad=False,
    )


def gfpgan_identity_arcface_resnet18(*args, **kwargs) -> Module:
    return gfpgan_arcface_resnet18(
        root_dir=kwargs.get("arcface_weights_dir", "_d/arcface")
    )


def identity_arcface_iresnet18(*args, **kwargs) -> Module:
    return arcface_iresnet18(
        root_dir=kwargs.get("arcface_weights_dir", "_d/infa_checkpoints/arcface"),
        requires_grad=False,
    )


def identity_arcface_iresnet34(*args, **kwargs) -> Module:
    return arcface_iresnet34(
        root_dir=kwargs.get("arcface_weights_dir", "_d/infa_checkpoints/arcface"),
        requires_grad=False,
    )


def identity_arcface_iresnet50(*args, **kwargs) -> Module:
    return arcface_iresnet50(
        root_dir=kwargs.get("arcface_weights_dir", "_d/infa_checkpoints/arcface"),
        requires_grad=False,
    )


def identity_arcface_iresnet100(*args, **kwargs) -> Module:
    return arcface_iresnet100(
        root_dir=kwargs.get("arcface_weights_dir", "_d/infa_checkpoints/arcface"),
        requires_grad=False,
    )


def components_gan_loss(*args, **kwargs) -> Module:
    return GANLoss(
        gan_type="vanilla",
        real_label_val=1.0,
        fake_label_val=0.0,
        loss_weight=1.0,
    )


LOSS_CRITERION_MAP: Dict[str, str] = {
    "loss__identity_arcface_resnet18": [gfpgan_identity_arcface_resnet18.__name__],
    "loss__identity_arcface_iresnet18": [identity_arcface_iresnet18.__name__],
    "loss__identity_arcface_iresnet34": [identity_arcface_iresnet34.__name__],
    "loss__identity_arcface_iresnet50": [identity_arcface_iresnet50.__name__],
    "loss__identity_arcface_iresnet100": [identity_arcface_iresnet100.__name__],
    "loss__identity_face_clip": [face_clip.__name__],
    "component__generator_left_eye": [components_gan_loss.__name__],
    "component__generator_right_eye": [components_gan_loss.__name__],
    "component__generator_mouth": [components_gan_loss.__name__],
    "component__disc_left_eye": [components_gan_loss.__name__],
    "component__disc_right_eye": [components_gan_loss.__name__],
    "component__disc_mouth": [components_gan_loss.__name__],
}
