import math

import torch
import torch.nn as nn
from torch.nn import functional as F

__all__ = [
    "U2Net",
    "U2NetV2",
    "U2NET_full",
    "U2NET_lite",
    "U2NET_full_encoder",
    "U2NET_full_decoder",
    "U2NET_decoder",
    "U2NETV2_full",
    "U2NETV2_lite",
]

from typing import Dict, List, Union

from inplace_abn import ABN, InPlaceABN, InPlaceABNSync


def _upsample_like(x: torch.Tensor, size):
    return nn.Upsample(size=size, mode="bilinear", align_corners=False)(x)


def upsample(x: torch.Tensor, scale_factor: int = 2):
    return F.interpolate(
        x, scale_factor=scale_factor, mode="bilinear", align_corners=False
    )


def _size_map(x: torch.Tensor, height: int):
    # {height: size} for Upsample
    size = list(x.shape[-2:])
    sizes: Dict[int, List[int]] = {}
    for h in range(1, height):
        sizes[h] = size
        size = [math.ceil(w / 2) for w in size]
    return sizes


class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dilate=1):
        super().__init__()

        self.conv_s1 = nn.Conv2d(
            in_ch, out_ch, 3, padding=1 * dilate, dilation=1 * dilate, bias=False
        )
        self.abn = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu_s1(self.abn(self.conv_s1(x)))


class REBNCONVSync(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dilate=1):
        super().__init__()

        self.conv_s1 = nn.Conv2d(
            in_ch, out_ch, 3, padding=1 * dilate, dilation=1 * dilate, bias=False
        )
        self.abn = nn.SyncBatchNorm(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu_s1(self.abn(self.conv_s1(x)))


class REAbnCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dilate=1):
        super().__init__()
        self.conv_s1 = nn.Conv2d(
            in_ch, out_ch, 3, padding=1 * dilate, dilation=1 * dilate, bias=False
        )
        self.abn = InPlaceABN(out_ch)

    def forward(self, x):
        return self.abn(self.conv_s1(x))


class REAbnCONVSync(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dilate=1):
        super().__init__()
        self.conv_s1 = nn.Conv2d(
            in_ch, out_ch, 3, padding=1 * dilate, dilation=1 * dilate, bias=False
        )
        self.abn = InPlaceABNSync(out_ch)

    def forward(self, x):
        return self.abn(self.conv_s1(x))


class RSU(nn.Module):
    def __init__(
        self, name, height, in_ch, mid_ch, out_ch, dilated=False, conv_module=REAbnCONV
    ):
        super(RSU, self).__init__()
        self.name = name
        self.height = height
        self.dilated = dilated
        self._make_layers(
            height, in_ch, mid_ch, out_ch, dilated, conv_module=conv_module
        )

    def _unet(self, x, sizes, height):
        if height < self.height:
            x1 = getattr(self, f"rebnconv{height}")(x)
            if not self.dilated and height < self.height - 1:
                x2 = self._unet(self.downsample(x1), sizes, height + 1)
            else:
                x2 = self._unet(x1, sizes, height + 1)

            x = getattr(self, f"rebnconv{height}d")(torch.cat((x2, x1), 1))
            return (
                _upsample_like(x, sizes[height - 1])
                if not self.dilated and height > 1
                else x  # U-Net like symmetric encoder-decoder structure
            )
        else:
            return getattr(self, f"rebnconv{height}")(x)

    def forward(self, x):
        sizes = _size_map(x, self.height)
        x = self.rebnconvin(x)

        return x + self._unet(x, sizes, 1)

    def _make_layers(
        self, height, in_ch, mid_ch, out_ch, dilated=False, conv_module=REBNCONV
    ):
        self.add_module("rebnconvin", conv_module(in_ch, out_ch))
        self.add_module("downsample", nn.MaxPool2d(2, stride=2, ceil_mode=True))

        self.add_module(f"rebnconv1", conv_module(out_ch, mid_ch))
        self.add_module(f"rebnconv1d", conv_module(mid_ch * 2, out_ch))

        for i in range(2, height):
            dilate = 1 if not dilated else 2 ** (i - 1)
            self.add_module(f"rebnconv{i}", conv_module(mid_ch, mid_ch, dilate=dilate))
            self.add_module(
                f"rebnconv{i}d", conv_module(mid_ch * 2, mid_ch, dilate=dilate)
            )

        dilate = 2 if not dilated else 2 ** (height - 1)
        self.add_module(f"rebnconv{height}", conv_module(mid_ch, mid_ch, dilate=dilate))


class U2NET(nn.Module):
    def __init__(self, cfgs, out_ch, use_abn=True):
        super(U2NET, self).__init__()
        self.out_ch = out_ch
        self._make_layers(cfgs, conv_module=(REAbnCONV if use_abn else REBNCONV))

    # side saliency map
    def unet(self, x, maps, sizes, height):
        if height < 6:
            x1 = getattr(self, f"stage{height}")(x)
            x2 = self.unet(getattr(self, "downsample")(x1), maps, sizes, height + 1)
            x = getattr(self, f"stage{height}d")(torch.cat((x2, x1), 1))
            self.side(x, height, maps, sizes)
            return _upsample_like(x, sizes[height - 1]) if height > 1 else x
        else:
            x = getattr(self, f"stage{height}")(x)
            self.side(x, height, maps, sizes)
            return _upsample_like(x, sizes[height - 1])

    def side(self, x, h, maps, sizes):
        # side output saliency map (before sigmoid)
        x = getattr(self, f"side{h}")(x)
        x = _upsample_like(x, sizes[1])
        maps.append(x)

    def fuse(self, maps):
        # fuse saliency probability maps
        maps.reverse()
        x = torch.cat(maps, 1)
        x = getattr(self, "outconv")(x)
        maps.insert(0, x)
        return maps

    def forward(self, x):
        sizes = _size_map(x, self.height)
        maps = []  # storage for maps
        self.unet(x, maps, sizes, 1)
        maps = self.fuse(maps)
        return maps[0]

    def _make_layers(self, cfgs, conv_module):
        self.height = int((len(cfgs) + 1) / 2)
        self.add_module("downsample", nn.MaxPool2d(2, stride=2, ceil_mode=True))
        for k, v in cfgs.items():
            # build rsu block
            self.add_module(k, RSU(v[0], *v[1], conv_module=conv_module))
            if v[2] > 0:
                # build side layer
                self.add_module(
                    f"side{v[0][-1]}", nn.Conv2d(v[2], self.out_ch, 3, padding=1)
                )
        # build fuse layer
        self.add_module(
            "outconv",
            nn.Conv2d(
                int(self.height * self.out_ch), self.out_ch, 1
            ),  # TODO: try bigger kernel size
        )


class U2NetV2(nn.Module):
    def __init__(
        self,
        cfgs,
        out_ch: int = 3,
        use_abn: bool = True,
        use_sync_bn: bool = False,
    ):
        super(U2NetV2, self).__init__()
        self.encoder = U2NetEncoder(
            cfgs["encoder"], use_abn=use_abn, use_sync_bn=use_sync_bn
        )
        self.decoder = U2NetDecoder(
            cfgs["decoder"], out_ch=out_ch, use_abn=use_abn, use_sync_bn=use_sync_bn
        )
        self.out_ch = out_ch

    def forward(self, x):
        encoder_features = self.encoder(x)
        return self.decoder(encoder_features)


class U2NetEncoder(nn.Module):
    def __init__(
        self,
        cfgs,
        use_abn: bool = True,
        use_sync_bn: bool = False,
    ):
        super(U2NetEncoder, self).__init__()
        self.cfgs: Dict[str, List] = cfgs
        if use_abn:
            if use_sync_bn:
                conv_module = REAbnCONVSync
            else:
                conv_module = REAbnCONV
        else:
            if use_sync_bn:
                conv_module = REBNCONVSync
            else:
                conv_module = REBNCONV
        self.__make_layers(conv_module=conv_module)

    def forward(self, x):
        features = []
        feat = x
        for i in range(1, self._height + 1):
            feat = getattr(self, f"stage{i}")(feat)
            features.append(feat)
            if i < self._height:
                feat = self.downsample(feat)
        return features

    def __make_layers(self, conv_module: nn.Module):
        self._height = len(self.cfgs)
        self.downsample = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        for k, v in self.cfgs.items():
            self.add_module(k, RSU(v[0], *v[1], conv_module=conv_module))


class U2NetDecoder(nn.Module):
    def __init__(
        self,
        cfgs,
        out_ch: int = 3,
        use_abn: bool = True,
        use_sync_bn: bool = False,
    ):
        super(U2NetDecoder, self).__init__()
        self.cfgs: Dict[str, List] = cfgs
        self.out_ch = out_ch
        if use_abn:
            if use_sync_bn:
                conv_module = REAbnCONVSync
            else:
                conv_module = REAbnCONV
        else:
            if use_sync_bn:
                conv_module = REBNCONVSync
            else:
                conv_module = REBNCONV
        self.__make_layers(conv_module=conv_module)

    def _side(self, x, level, maps, scale_factor):
        # side output saliency map
        # print(f"_side: 1 : {x.shape = }, {level = }, {scale_factor = }")
        s_map = getattr(self, f"side{level}")(x)
        if scale_factor > 1:
            s_map = upsample(s_map, scale_factor=scale_factor)
        # print(f"_side: 2 : {s_map.shape = }")
        maps.append(s_map)
        return maps

    def _fuse(self, maps):
        # fuse saliency probability maps
        maps.reverse()
        x = torch.cat(maps, 1)
        # print(f"_fuse:: {x.shape = }")
        return self.outconv(x)

    def forward(self, inp_feature_lst):
        """
        inp_feature_lst: list of features from encoder
        """
        maps = []
        feature_prev = None
        # print("=" * 50)
        # for i, f in enumerate(inp_feature_lst):
        #     print(f"{i = }, {f.shape = }")
        for i, inp_feature in zip(
            reversed(range(1, self._height + 1)), reversed(inp_feature_lst)
        ):
            # print(f"{i = }")
            # print(f"{inp_feature.shape = }")
            if i < self._height:
                res = getattr(self, f"stage{i}d")(
                    torch.cat((inp_feature, feature_prev), 1)
                )
            else:
                res = inp_feature
            # print(f"{res.shape = }")
            maps = self._side(
                res,
                level=i,
                maps=maps,
                scale_factor=2 ** (i - 1),
            )
            if i > 1:
                feature_prev = upsample(res, scale_factor=2)
                # print(f"{feature_prev.shape = }")

        out = self._fuse(maps)
        # print(f"{out.shape = }")
        return out

    def __make_layers(self, conv_module):
        self._height = len(self.cfgs)
        for k, v in self.cfgs.items():
            # RSU block
            if v[1] is not None:
                self.add_module(k, RSU(v[0], *v[1], conv_module=conv_module))
            # side layer
            self.add_module(
                f"side{v[0][-1]}", nn.Conv2d(v[2], self.out_ch, 3, padding=1)
            )
        # fuse layer
        self.outconv = nn.Conv2d(int(self._height * self.out_ch), self.out_ch, 1)


def U2NET_encoder(in_ch: List[int], out_ch=1, use_abn=True):
    cfg = {
        "stage1": ["En_1", (7, in_ch[0], in_ch[1] // 2, in_ch[1]), -1],
        "stage2": ["En_2", (6, in_ch[1], in_ch[2] // 4, in_ch[2]), -1],
        "stage3": ["En_3", (5, in_ch[2], in_ch[3] // 2, in_ch[3]), -1],
        "stage4": ["En_4", (4, in_ch[3], in_ch[4] // 4, in_ch[4]), -1],
        "stage5": ["En_5", (4, in_ch[4], in_ch[4] // 2, in_ch[4], True), -1],
        "stage6": ["En_6", (4, in_ch[4], in_ch[4] // 2, in_ch[4], True), in_ch[4]],
    }
    return U2NetEncoder(cfg, use_abn=use_abn)


def U2NET_full_encoder(in_ch: int = 3, out_ch: int = 1, use_abn: bool = True):
    cfg = {
        "stage1": ["En_1", (7, in_ch, 32, 64), -1],
        "stage2": ["En_2", (6, 64, 32, 128), -1],
        "stage3": ["En_3", (5, 128, 64, 256), -1],
        "stage4": ["En_4", (4, 256, 128, 512), -1],
        "stage5": ["En_5", (4, 512, 256, 512, True), -1],
        "stage6": ["En_6", (4, 512, 256, 512, True), 512],
    }
    return U2NetEncoder(cfg, use_abn=use_abn)


def U2NET_lite_encoder(in_ch: int = 3, out_ch: int = 1, use_abn: bool = True):
    cfg = {
        "stage1": ["En_1", (7, in_ch, 16, 64), -1],
        "stage2": ["En_2", (6, 64, 16, 64), -1],
        "stage3": ["En_3", (5, 64, 16, 64), -1],
        "stage4": ["En_4", (4, 64, 16, 64), -1],
        "stage5": ["En_5", (4, 64, 16, 64, True), -1],
        "stage6": ["En_6", (4, 64, 16, 64, True), 64],
    }
    return U2NetEncoder(cfg, use_abn=use_abn)


def U2NET_decoder(in_ch: List[int], out_ch=3, use_abn=True):
    cfg = {
        "stage6": ["En_6", None, in_ch[0]],
        "stage5d": ["De_5", (4, in_ch[0] + in_ch[1], 256, in_ch[1], True), in_ch[1]],
        "stage4d": ["De_4", (4, in_ch[1] + in_ch[2], 128, in_ch[2]), in_ch[2]],
        "stage3d": ["De_3", (5, in_ch[2] + in_ch[3], 64, in_ch[3]), in_ch[3]],
        "stage2d": ["De_2", (6, in_ch[3] + in_ch[4], 32, in_ch[4]), in_ch[4]],
        "stage1d": ["De_1", (7, in_ch[4] + in_ch[5], 16, in_ch[5]), in_ch[5]],
    }
    return U2NetDecoder(cfg, out_ch=out_ch, use_abn=use_abn)


def U2NET_full_decoder(in_ch: int = 3, out_ch=3, use_abn=True):
    cfg = {
        "stage6": ["En_6", None, 512],
        "stage5d": ["De_5", (4, 1024, 256, 512, True), 512],
        "stage4d": ["De_4", (4, 1024, 128, 256), 256],
        "stage3d": ["De_3", (5, 512, 64, 128), 128],
        "stage2d": ["De_2", (6, 256, 32, 64), 64],
        "stage1d": ["De_1", (7, 128, 16, 64), 64],
    }
    return U2NetDecoder(cfg, out_ch=out_ch, use_abn=use_abn)


def U2NET_lite_decoder(in_ch: int = 3, out_ch=3, use_abn=True):
    cfg = {
        "stage6": ["En_6", None, 64],
        "stage5d": ["De_5", (4, 128, 16, 64, True), 64],
        "stage4d": ["De_4", (4, 128, 16, 64), 64],
        "stage3d": ["De_3", (5, 128, 16, 64), 64],
        "stage2d": ["De_2", (6, 128, 16, 64), 64],
        "stage1d": ["De_1", (7, 128, 16, 64), 64],
    }
    return U2NetDecoder(cfg, out_ch=out_ch, use_abn=use_abn)


def U2NETV2_full(in_ch=3, out_ch=1, use_abn: bool = True, use_sync_bn: bool = False):
    enc_cfg = {
        "stage1": ["En_1", (7, in_ch, 32, 64), -1],
        "stage2": ["En_2", (6, 64, 32, 128), -1],
        "stage3": ["En_3", (5, 128, 64, 256), -1],
        "stage4": ["En_4", (4, 256, 128, 512), -1],
        "stage5": ["En_5", (4, 512, 256, 512, True), -1],
        "stage6": ["En_6", (4, 512, 256, 512, True), 512],
    }
    dec_cfg = {
        "stage6": ["En_6", None, 512],
        "stage5d": ["De_5", (4, 1024, 256, 512, True), 512],
        "stage4d": ["De_4", (4, 1024, 128, 256), 256],
        "stage3d": ["De_3", (5, 512, 64, 128), 128],
        "stage2d": ["De_2", (6, 256, 32, 64), 64],
        "stage1d": ["De_1", (7, 128, 16, 64), 64],
    }
    full = {"encoder": enc_cfg, "decoder": dec_cfg}
    return U2NetV2(cfgs=full, out_ch=out_ch, use_abn=use_abn, use_sync_bn=use_sync_bn)


def U2NET_full(in_ch=3, out_ch=1, use_abn=True):
    full = {
        # cfgs for building RSUs and sides
        # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
        "stage1": ["En_1", (7, in_ch, 32, 64), -1],
        "stage2": ["En_2", (6, 64, 32, 128), -1],
        "stage3": ["En_3", (5, 128, 64, 256), -1],
        "stage4": ["En_4", (4, 256, 128, 512), -1],
        "stage5": ["En_5", (4, 512, 256, 512, True), -1],
        "stage6": ["En_6", (4, 512, 256, 512, True), 512],
        "stage5d": ["De_5", (4, 1024, 256, 512, True), 512],
        "stage4d": ["De_4", (4, 1024, 128, 256), 256],
        "stage3d": ["De_3", (5, 512, 64, 128), 128],
        "stage2d": ["De_2", (6, 256, 32, 64), 64],
        "stage1d": ["De_1", (7, 128, 16, 64), 64],
    }
    return U2NET(cfgs=full, out_ch=out_ch, use_abn=use_abn)


def U2NETV2_lite(in_ch=3, out_ch=3, use_abn: bool = True, use_sync_bn: bool = False):
    enc_cfg = {
        "stage1": ["En_1", (7, in_ch, 16, 64), -1],
        "stage2": ["En_2", (6, 64, 16, 64), -1],
        "stage3": ["En_3", (5, 64, 16, 64), -1],
        "stage4": ["En_4", (4, 64, 16, 64), -1],
        "stage5": ["En_5", (4, 64, 16, 64, True), -1],
        "stage6": ["En_6", (4, 64, 16, 64, True), 64],
    }
    dec_cfg = {
        "stage6": ["En_6", None, 64],
        "stage5d": ["De_5", (4, 128, 16, 64, True), 64],
        "stage4d": ["De_4", (4, 128, 16, 64), 64],
        "stage3d": ["De_3", (5, 128, 16, 64), 64],
        "stage2d": ["De_2", (6, 128, 16, 64), 64],
        "stage1d": ["De_1", (7, 128, 16, 64), 64],
    }
    full = {"encoder": enc_cfg, "decoder": dec_cfg}
    return U2NetV2(cfgs=full, out_ch=out_ch, use_abn=use_abn, use_sync_bn=use_sync_bn)


def U2NET_lite(in_ch=3, out_ch=1, use_abn=True):
    lite = {
        # cfgs for building RSUs and sides
        # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
        "stage1": ["En_1", (7, in_ch, 16, 64), -1],
        "stage2": ["En_2", (6, 64, 16, 64), -1],
        "stage3": ["En_3", (5, 64, 16, 64), -1],
        "stage4": ["En_4", (4, 64, 16, 64), -1],
        "stage5": ["En_5", (4, 64, 16, 64, True), -1],
        "stage6": ["En_6", (4, 64, 16, 64, True), 64],
        "stage5d": ["De_5", (4, 128, 16, 64, True), 64],
        "stage4d": ["De_4", (4, 128, 16, 64), 64],
        "stage3d": ["De_3", (5, 128, 16, 64), 64],
        "stage2d": ["De_2", (6, 128, 16, 64), 64],
        "stage1d": ["De_1", (7, 128, 16, 64), 64],
    }
    return U2NET(cfgs=lite, out_ch=out_ch, use_abn=use_abn)


if __name__ == "__main__":
    x = torch.randn((1, 3, 160, 160))
    enc = U2NET_full_encoder(in_ch=3, out_ch=1, use_abn=True)
    dec = U2NET_full_decoder(in_ch=3, out_ch=3, use_abn=True)

    enc_res = enc(x)
    print(f"Encoder res len: {len(enc_res) = }")
    for i, er in enumerate(enc_res):
        print(f"{i = }:: {er.shape = }")

    dec_res = dec(enc_res)
    print(f"Decoder res len: {len(dec_res) = }")
    for i, dr in enumerate(dec_res):
        print(f"{i = }:: {dr.shape = }")
