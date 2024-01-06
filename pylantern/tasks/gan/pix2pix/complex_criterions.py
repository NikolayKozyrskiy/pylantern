from typing import Dict

import lpips
import vision_aided_loss as visal
from basicsr.losses.basic_loss import PerceptualLoss
from basicsr.losses.gan_loss import GANLoss
from dreamsim import dreamsim
from torch.nn import Module

from pylantern.common.criterions import LaplacianLoss, MSDSSIMLoss, SobelOperatorLoss
from pylantern.common.utils import get_device
from pylantern.model_zoo.models import (
    arcface_iresnet18,
    arcface_iresnet34,
    arcface_iresnet50,
    arcface_iresnet100,
    face_clip_vitb16,
    gfpgan_arcface_resnet18,
)
from pylantern.tasks.gan.common.nn.perceptual import VGG19Loss


def components_gan_loss(*args, **kwargs) -> Module:
    return GANLoss(
        gan_type="vanilla",
        real_label_val=1.0,
        fake_label_val=0.0,
        loss_weight=1.0,
    )


def wgan_softplus_loss(*args, **kwargs) -> Module:
    return GANLoss(
        gan_type="wgan_softplus",
        real_label_val=1.0,
        fake_label_val=0.0,
        loss_weight=0.1,
    )


def ms_dssim(*args, **kwargs) -> Module:
    return MSDSSIMLoss(data_range=1.0, reduction="mean")


def vgg19_perceptual(*args, **kwargs) -> Module:
    return VGG19Loss(
        style_weight=0.0,
        normalize_inputs=True,
        resize_inputs=False,
        reduction="mean",
    )


def vgg19_perceptual_basicsr(*args, **kwargs) -> Module:
    return PerceptualLoss(
        layer_weights={
            "conv1_2": 0.1,
            "conv2_2": 0.1,
            "conv3_4": 1,
            "conv4_4": 1,
            "conv5_4": 1,
        },
        vgg_type="vgg19",
        use_input_norm=True,
        range_norm=True,
        perceptual_weight=1.0,
        style_weight=50.0,
        criterion="l1",
    )


def vgg19_perceptual_and_style(*args, **kwargs) -> Module:
    return VGG19Loss(
        style_weight=kwargs.get("vgg_style_weight", 30.0),
        normalize_inputs=True,
        resize_inputs=False,
        reduction="mean",
    )


def lpips_alex_perceptual(*args, **kwargs) -> Module:
    return lpips.LPIPS(net="alex")


def lpips_vgg16_perceptual(*args, **kwargs) -> Module:
    return lpips.LPIPS(net="vgg16")


def lpips_squeeze_perceptual(*args, **kwargs) -> Module:
    return lpips.LPIPS(net="squeeze")


def dreamsim_ensemble(*args, **kwargs) -> Module:
    return dreamsim(
        dreamsim_type="ensemble",
        pretrained=True,
        normalize_embeds=True,
        device=kwargs.get("device", get_device()),
        cache_dir=kwargs.get("dreamsim_weights_dir", "_d/dreamsim"),
    )[0]


def sobel(*args, **kwargs) -> Module:
    return SobelOperatorLoss(
        device=kwargs.get("device", get_device()), reduction="mean"
    )


def laplacian_pyramid(*args, **kwargs) -> Module:
    return LaplacianLoss(max_levels=kwargs.get("max_levels", 3))


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


def vision_aided_discriminator(*args, **kwargs) -> Module:
    discriminator = visal.Discriminator(
        cv_type="clip",
        loss_type="multilevel_sigmoid_s",
        diffaug=False,
        device=kwargs.get("device", get_device()),
    )
    discriminator.cv_ensemble.requires_grad_(False)
    return discriminator


LOSS_CRITERION_MAP: Dict[str, str] = {
    "gan__vision_aided_discriminator": [vision_aided_discriminator.__name__],
    "gan__vision_aided_generator": [vision_aided_discriminator.__name__],
    "gan__temporal_laplacian_pyramid": [laplacian_pyramid.__name__],
    "gan__temporal_ms_dssim": [ms_dssim.__name__],
    "loss__vgg_perceptual": [vgg19_perceptual.__name__],
    "loss__vgg19_perceptual_and_style": [vgg19_perceptual_and_style.__name__],
    "loss__vgg19_perceptual_basicsr": [vgg19_perceptual_basicsr.__name__],
    "loss__lpips_perceptual": [lpips_alex_perceptual.__name__],
    "loss__dreamsim_cos_ensemble_perceptual": [dreamsim_ensemble.__name__],
    "loss__identity_arcface_resnet18": [gfpgan_identity_arcface_resnet18.__name__],
    "loss__identity_arcface_iresnet18": [identity_arcface_iresnet18.__name__],
    "loss__identity_arcface_iresnet34": [identity_arcface_iresnet34.__name__],
    "loss__identity_arcface_iresnet50": [identity_arcface_iresnet50.__name__],
    "loss__identity_arcface_iresnet100": [identity_arcface_iresnet100.__name__],
    "loss__identity_face_clip": [face_clip.__name__],
    "loss__recon_ms_dssim": [ms_dssim.__name__],
    "loss__sobel": [sobel.__name__],
    "component__generator_left_eye": [components_gan_loss.__name__],
    "component__generator_right_eye": [components_gan_loss.__name__],
    "component__generator_mouth": [components_gan_loss.__name__],
    "component__disc_left_eye": [components_gan_loss.__name__],
    "component__disc_right_eye": [components_gan_loss.__name__],
    "component__disc_mouth": [components_gan_loss.__name__],
    "gan__generator": [wgan_softplus_loss.__name__],
}
