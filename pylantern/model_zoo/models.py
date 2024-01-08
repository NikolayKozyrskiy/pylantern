import os
from typing import List

import torch
from torch.nn import Module

from pylantern.common.utils.module import (
    deep_network_interpolation,
    model_eval,
    remove_module_from_state_dict,
)


def face_clip_vitb16(root_dir: str, requires_grad: bool = False) -> Module:
    from .clip import load_clip

    model, _ = load_clip("ViT-B/16", device="cpu")
    farl_state = torch.load(
        os.path.join(root_dir, "FaRL-Base-Patch16-LAIONFace20M-ep64.pth")
    )
    model.load_state_dict(farl_state["state_dict"], strict=False)
    model.eval().float()
    if not requires_grad:
        for param in model.parameters():
            param.requires_grad = False
    return model


def arcface_iresnet18(root_dir: str, requires_grad: bool = False) -> Module:
    from .arcface import iresnet18

    model = iresnet18()
    checkpoint = torch.load(
        os.path.join(root_dir, "arcface_fp16_r18.pth"), map_location="cpu"
    )
    return model_eval(model=model, state_dict=checkpoint, requires_grad=requires_grad)


def arcface_iresnet34(root_dir: str, requires_grad: bool = False) -> Module:
    from .arcface import iresnet34

    model = iresnet34()
    checkpoint = torch.load(
        os.path.join(root_dir, "arcface_fp16_r34.pth"), map_location="cpu"
    )
    return model_eval(model=model, state_dict=checkpoint, requires_grad=requires_grad)


def arcface_iresnet50(root_dir: str, requires_grad: bool = False) -> Module:
    from .arcface import iresnet50

    model = iresnet50()
    checkpoint = torch.load(
        os.path.join(root_dir, "arcface_fp16_r50.pth"), map_location="cpu"
    )
    return model_eval(model=model, state_dict=checkpoint, requires_grad=requires_grad)


def arcface_iresnet100(root_dir: str, requires_grad: bool = False) -> Module:
    from .arcface import iresnet100

    model = iresnet100()
    checkpoint = torch.load(
        os.path.join(root_dir, "arcface_fp16_r100.pth"), map_location="cpu"
    )
    return model_eval(model=model, state_dict=checkpoint, requires_grad=requires_grad)


def codeformer(root_dir: str) -> Module:
    from basicsr.utils.download_util import load_file_from_url

    from .codeformer import CodeFormer

    model = CodeFormer(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=["32", "64", "128", "256"],
    )

    url = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
    ckpt_path = load_file_from_url(
        url=url,
        model_dir=os.path.join(root_dir, "CodeFormer"),
        progress=True,
        file_name=None,
    )
    checkpoint = torch.load(ckpt_path)["params_ema"]
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def gfpgan_arcface_resnet18(root_dir: str, requires_grad: bool = False) -> Module:
    from .gfpgan import ResNetArcFace

    model = ResNetArcFace(block="IRBlock", layers=[2, 2, 2, 2], use_se=False)
    checkpoint = torch.load(
        os.path.join(root_dir, "arcface_resnet18.pth"), map_location="cpu"
    )
    checkpoint = remove_module_from_state_dict(checkpoint)
    return model_eval(model=model, state_dict=checkpoint, requires_grad=requires_grad)


def gfpgan_v14(root_dir: str) -> Module:
    model_name = "GFPGANv1.4"
    url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
    return _gfpgan_clean(root_dir=root_dir, model_path=url)


def gfpgan_v13(root_dir: str) -> Module:
    model_name = "GFPGANv1.3"
    url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
    return _gfpgan_clean(root_dir=root_dir, model_path=url)


def gfpgan_v12(root_dir: str) -> Module:
    model_name = "GFPGANCleanv1-NoCE-C2"
    url = "https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth"
    return _gfpgan_clean(root_dir=root_dir, model_path=url)


def restoreformer(root_dir: str) -> Module:
    model_name = "RestoreFormer"
    url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth"
    return _restoreformer(root_dir=root_dir, model_path=url)


def gfpgan_v1_old(root_dir: str) -> Module:
    model_name = "GFPGANv1"
    url = "https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth"
    return _gfpgan_old(root_dir=root_dir, model_path=url)


def _gfpgan_clean(root_dir: str, model_path: str) -> Module:
    from .gfpgan import GFPGANv1Clean

    channel_multiplier = 2
    gfpgan = GFPGANv1Clean(
        out_size=512,
        num_style_feat=512,
        channel_multiplier=channel_multiplier,
        decoder_load_path=None,
        fix_decoder=False,
        num_mlp=8,
        input_is_latent=True,
        different_w=True,
        narrow=1,
        sft_half=True,
    )
    return _load_tencent_model(gfpgan, root_dir=root_dir, model_path=model_path)


def _gfpgan_old(root_dir: str, model_path: str) -> Module:
    from .gfpgan import GFPGANv1

    channel_multiplier = 1
    gfpgan = GFPGANv1(
        out_size=512,
        num_style_feat=512,
        channel_multiplier=channel_multiplier,
        decoder_load_path=None,
        fix_decoder=True,
        num_mlp=8,
        input_is_latent=True,
        different_w=True,
        narrow=1,
        sft_half=True,
    )
    return _load_tencent_model(gfpgan, root_dir=root_dir, model_path=model_path)


def _restoreformer(root_dir: str, model_path: str) -> Module:
    from .restoreformer import RestoreFormer

    return _load_tencent_model(
        RestoreFormer(), root_dir=root_dir, model_path=model_path
    )


def _load_tencent_model(model: Module, root_dir: str, model_path: str) -> Module:
    from basicsr.utils.download_util import load_file_from_url

    if model_path.startswith("https://"):
        model_path = load_file_from_url(
            url=model_path,
            model_dir=os.path.join(root_dir, "gfpgan/weights"),
            progress=True,
            file_name=None,
        )
    loadnet = torch.load(model_path)
    if "params_ema" in loadnet:
        keyname = "params_ema"
    else:
        keyname = "params"
    model.load_state_dict(loadnet[keyname], strict=True)
    model.eval()

    return model


def RealESRGAN_x4plus(root_dir: str) -> Module:
    from basicsr.archs.rrdbnet_arch import RRDBNet

    model_name = "RealESRGAN_x4plus"
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )
    netscale = 4
    file_url = [
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    ]
    return _load_real_esrgan(
        model=model,
        model_name=model_name,
        root_dir=root_dir,
        file_urls=file_url,
        denoise_strength=1,
    )


def RealESRNet_x4plus(root_dir: str) -> Module:
    from basicsr.archs.rrdbnet_arch import RRDBNet

    model_name = "RealESRNet_x4plus"
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )
    netscale = 4
    file_url = [
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
    ]
    return _load_real_esrgan(
        model=model,
        model_name=model_name,
        root_dir=root_dir,
        file_urls=file_url,
        denoise_strength=1,
    )


def RealESRGAN_x4plus_anime_6B(root_dir: str) -> Module:
    from basicsr.archs.rrdbnet_arch import RRDBNet

    model_name = "RealESRGAN_x4plus_anime_6B"
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=6,
        num_grow_ch=32,
        scale=4,
    )
    netscale = 4
    file_url = [
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
    ]
    return _load_real_esrgan(
        model=model,
        model_name=model_name,
        root_dir=root_dir,
        file_urls=file_url,
        denoise_strength=1,
    )


def RealESRGAN_x2plus(root_dir: str) -> Module:
    from basicsr.archs.rrdbnet_arch import RRDBNet

    model_name = "RealESRGAN_x2plus"
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )
    netscale = 2
    file_url = [
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    ]
    return _load_real_esrgan(
        model=model,
        model_name=model_name,
        root_dir=root_dir,
        file_urls=file_url,
        denoise_strength=1,
    )


def realesr_animevideov3(root_dir: str) -> Module:
    from .vgg import SRVGGNetCompact

    model_name = "realesr-animevideov3"
    model = SRVGGNetCompact(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_conv=16,
        upscale=4,
        act_type="prelu",
    )
    netscale = 4
    file_url = [
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
    ]
    return _load_real_esrgan(
        model=model,
        model_name=model_name,
        root_dir=root_dir,
        file_urls=file_url,
        denoise_strength=1,
    )


def realesr_general_x4v3(root_dir: str, denoise_strength: int = 1):
    from .vgg import SRVGGNetCompact

    model_name = "realesr-general-x4v3"
    model = SRVGGNetCompact(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_conv=32,
        upscale=4,
        act_type="prelu",
    )
    netscale = 4
    file_url = [
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    ]
    return _load_real_esrgan(
        model=model,
        model_name=model_name,
        root_dir=root_dir,
        file_urls=file_url,
        denoise_strength=denoise_strength,
    )


def _load_real_esrgan(
    model: Module,
    model_name: str,
    root_dir: str,
    file_urls: List[str],
    denoise_strength: int = 1,
):
    from basicsr.utils.download_util import load_file_from_url

    for url in file_urls:
        model_path = load_file_from_url(
            url=url,
            model_dir=os.path.join(root_dir, "real_esrgan/weights"),
            progress=True,
            file_name=None,
        )
    if model_name == "realesr-general-x4v3" and denoise_strength != 1:
        # use dni to control the denoise strength
        wdn_model_path = model_path.replace(
            "realesr-general-x4v3", "realesr-general-wdn-x4v3"
        )
        dni_weight = [denoise_strength, 1 - denoise_strength]
        state_dict = deep_network_interpolation(model_path, wdn_model_path, dni_weight)
    else:
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))

    if "params_ema" in state_dict:
        keyname = "params_ema"
    else:
        keyname = "params"
    model.load_state_dict(state_dict[keyname], strict=True)
    model.eval()
    return model


def resnet18_small(num_classes: int, requires_grad: bool = False) -> Module:
    from .resnet import resnet18_small as _resnet18_small

    return model_eval(
        model=_resnet18_small(num_classes=num_classes), requires_grad=requires_grad
    )


def resnet34_small(num_classes: int, requires_grad: bool = False) -> Module:
    from .resnet import resnet34_small as _resnet34_small

    return model_eval(
        model=_resnet34_small(num_classes=num_classes), requires_grad=requires_grad
    )


def resnet50_small(num_classes: int, requires_grad: bool = False) -> Module:
    from .resnet import resnet50_small as _resnet50_small

    return model_eval(
        model=_resnet50_small(num_classes=num_classes), requires_grad=requires_grad
    )


def resnet101_small(num_classes: int, requires_grad: bool = False) -> Module:
    from .resnet import resnet101_small as _resnet101_small

    return model_eval(
        model=_resnet101_small(num_classes=num_classes), requires_grad=requires_grad
    )


def resnet152_small(num_classes: int, requires_grad: bool = False) -> Module:
    from .resnet import resnet152_small as _resnet152_small

    return model_eval(
        model=_resnet152_small(num_classes=num_classes), requires_grad=requires_grad
    )
