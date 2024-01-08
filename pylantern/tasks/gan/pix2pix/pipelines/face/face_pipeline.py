from functools import cached_property
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
from kornia.geometry.transform import warp_affine
from matches.shortcuts.dag import graph_node
from torch import Tensor, nn
from torchvision.ops import roi_align

from pylantern.common.constants import ARCFACE_DST_KPS
from pylantern.common.nn.functional import atm_arcface
from pylantern.common.utils import to_device
from pylantern.tasks.gan.pix2pix.pipelines.general.gfpgan_pipeline import GFPGANPipeline
from pylantern.tasks.gan.pix2pix.pipelines.general.pix2pixhd_pipeline import (
    Pix2PixHDPipeline,
)
from pylantern.tasks.gan.pix2pix.pipelines.pipeline import BasePix2PixPipeline

if TYPE_CHECKING:
    from pylantern.model_zoo.gfpgan import (
        FacialComponentDiscriminator,
        StyleGAN2Discriminator,
    )
    from pylantern.model_zoo.pix2pix import MultiscaleDiscriminator
    from pylantern.tasks.gan.pix2pix.configs import (
        FaceGFPGANConfig,
        FacePix2PixConfig,
        FacePix2PixHDConfig,
        FaceSpadeConfig,
    )


class FacePix2PixPipeline(BasePix2PixPipeline):
    def __init__(
        self,
        config: "FacePix2PixConfig",
        device: Union[str, torch.device],
        discriminator_left_eye: Optional["FacialComponentDiscriminator"] = None,
        discriminator_right_eye: Optional["FacialComponentDiscriminator"] = None,
        discriminator_mouth: Optional["FacialComponentDiscriminator"] = None,
    ):
        self.config: "FacePix2PixConfig" = config
        self.discriminator_left_eye = discriminator_left_eye
        self.discriminator_right_eye = discriminator_right_eye
        self.discriminator_mouth = discriminator_mouth

        self._arcface_dst_kps = (
            torch.tensor(ARCFACE_DST_KPS, dtype=torch.float64).to(device)
            if any(
                k in config.attrs_to_load.keys()
                for k in ("kps_5_buffalo_l_dst", "kps_5_buffalo_l_src")
            )
            else None
        )

    @cached_property
    def eye_roi_size(self) -> int:
        return int(round(self.config.eye_roi_relative_size * self.config.image_size[0]))

    @cached_property
    def mouth_roi_size(self) -> int:
        return int(
            round(self.config.mouth_roi_relative_size * self.config.image_size[0])
        )

    @graph_node
    def kps_5_buffalo_l_dst(self) -> Tensor:
        return self.batch["kps_5_buffalo_l_dst"]

    @graph_node
    def kps_5_buffalo_l_src(self) -> Tensor:
        return self.batch["kps_5_buffalo_l_src"]

    @graph_node
    def affine_transform_matrix_arcface_dst(self, image_size: int) -> Tensor:
        atm = atm_arcface(
            src_points=self.kps_5_buffalo_l_dst(),
            dst_points=self._arcface_dst_kps,
            image_size=image_size,
        )
        return atm

    @graph_node
    def dst_img_arcface_aligned_normalized_112(self) -> Tensor:
        img = self.image_dst_normalized()
        atm = self.affine_transform_matrix_arcface_dst(image_size=112)
        img_aligned = warp_affine(src=img, M=atm, dsize=(112, 112))
        return img_aligned

    @graph_node
    def get_predicted_image_arcface_aligned_normalized_112(self) -> Tensor:
        img = self.get_predicted_image()
        atm = self.affine_transform_matrix_arcface_dst(image_size=112)
        img_aligned = warp_affine(src=img, M=atm, dsize=(112, 112))
        return img_aligned

    @graph_node
    def dst_img_arcface_aligned_scaled_224(self) -> Tensor:
        img = self.image_dst_scaled()
        atm = self.affine_transform_matrix_arcface_dst(image_size=224)
        img_aligned = warp_affine(src=img, M=atm, dsize=(224, 224))
        return img_aligned

    @graph_node
    def get_predicted_image_arcface_aligned_destandardized_224(self) -> Tensor:
        img = self.get_predicted_image_destandardized()
        atm = self.affine_transform_matrix_arcface_dst(image_size=224)
        img_aligned = warp_affine(src=img, M=atm, dsize=(224, 224))
        return img_aligned

    @graph_node
    def eyes_roi_bboxes(self) -> Tensor:
        eyes_roi_bboxes = []
        loc_left_eyes = self.batch["left_eye_bbox"]
        loc_right_eyes = self.batch["right_eye_bbox"]
        for b in range(loc_left_eyes.size(0)):
            bbox = torch.stack([loc_left_eyes[b, :], loc_right_eyes[b, :]], dim=0)
            eyes_roi_bboxes.append(
                torch.cat(
                    [
                        loc_left_eyes.new_full((2, 1), b),
                        bbox,
                    ],
                    dim=-1,
                )
            )
        eyes_roi_bboxes = torch.cat(eyes_roi_bboxes, dim=0).to(loc_left_eyes)
        return eyes_roi_bboxes

    @graph_node
    def eyes_roi_image_dst_normalized(self) -> Tuple[Tensor, Tensor]:
        eyes = roi_align(
            input=self.image_dst_normalized(),
            boxes=self.eyes_roi_bboxes(),
            output_size=self.eye_roi_size,
        )
        return eyes[0::2, :, :, :], eyes[1::2, :, :, :]

    @graph_node
    def left_eye_roi_image_dst_normalized(self) -> Tensor:
        return self.eyes_roi_image_dst_normalized()[0]

    @graph_node
    def right_eye_roi_image_dst_normalized(self) -> Tensor:
        return self.eyes_roi_image_dst_normalized()[1]

    @graph_node
    @torch.no_grad()
    def left_eye_roi_image_dst_destandardized(self) -> Tensor:
        img = self.left_eye_roi_image_dst_normalized()
        res = img * self._std.expand_as(img)
        res = res + self._mean.expand_as(res)
        return res

    @graph_node
    @torch.no_grad()
    def right_eye_roi_image_dst_destandardized(self) -> Tensor:
        img = self.right_eye_roi_image_dst_normalized()
        res = img * self._std.expand_as(img)
        res = res + self._mean.expand_as(res)
        return res

    @graph_node
    def eyes_roi_image_generated(self) -> Tuple[Tensor, Tensor]:
        eyes = roi_align(
            input=self.get_predicted_image(),
            boxes=self.eyes_roi_bboxes(),
            output_size=self.eye_roi_size,
        )
        return eyes[0::2, :, :, :], eyes[1::2, :, :, :]

    @graph_node
    def left_eye_roi_image_generated(self) -> Tensor:
        return self.eyes_roi_image_generated()[0]

    @graph_node
    def right_eye_roi_image_generated(self) -> Tensor:
        return self.eyes_roi_image_generated()[1]

    @graph_node
    @torch.no_grad()
    def left_eye_roi_image_generated_destandardized(self) -> Tensor:
        img = self.left_eye_roi_image_generated()
        res = img * self._std.expand_as(img)
        res = res + self._mean.expand_as(res)
        return res

    @graph_node
    @torch.no_grad()
    def right_eye_roi_image_generated_destandardized(self) -> Tensor:
        img = self.right_eye_roi_image_generated()
        res = img * self._std.expand_as(img)
        res = res + self._mean.expand_as(res)
        return res

    @graph_node
    def mouth_roi_bboxes(self) -> Tensor:
        mouth_roi_bboxes = []
        loc_mouth = self.batch["mouth_bbox"]
        for b in range(loc_mouth.size(0)):
            mouth_roi_bboxes.append(
                torch.cat(
                    [
                        loc_mouth.new_full((1, 1), b),
                        loc_mouth[b : b + 1, :],
                    ],
                    dim=-1,
                )
            )
        mouth_roi_bboxes = torch.cat(mouth_roi_bboxes, dim=0).to(loc_mouth)
        return mouth_roi_bboxes

    @graph_node
    def mouth_roi_image_dst_normalized(self) -> Tensor:
        return roi_align(
            input=self.image_dst_normalized(),
            boxes=self.mouth_roi_bboxes(),
            output_size=self.mouth_roi_size,
        )

    @graph_node
    def mouth_roi_image_generated(self) -> Tensor:
        return roi_align(
            input=self.get_predicted_image(),
            boxes=self.mouth_roi_bboxes(),
            output_size=self.mouth_roi_size,
        )

    @graph_node
    def discriminator_left_eye_fake(self) -> Tuple[Tensor, List[Tensor]]:
        out, out_feats = self.discriminator_left_eye(
            self.left_eye_roi_image_generated(), return_feats=True
        )
        return out, out_feats

    @graph_node
    def discriminator_right_eye_fake(self) -> Tuple[Tensor, List[Tensor]]:
        out, out_feats = self.discriminator_right_eye(
            self.right_eye_roi_image_generated(), return_feats=True
        )
        return out, out_feats

    @graph_node
    def discriminator_mouth_fake(self) -> Tuple[Tensor, List[Tensor]]:
        out, out_feats = self.discriminator_mouth(
            self.mouth_roi_image_generated(), return_feats=True
        )
        return out, out_feats

    @graph_node
    def discriminator_left_eye_dst(self) -> Tuple[Tensor, List[Tensor]]:
        out, out_feats = self.discriminator_left_eye(
            self.left_eye_roi_image_dst_normalized(), return_feats=True
        )
        return out, out_feats

    @graph_node
    def discriminator_right_eye_dst(self) -> Tuple[Tensor, List[Tensor]]:
        out, out_feats = self.discriminator_right_eye(
            self.right_eye_roi_image_dst_normalized(), return_feats=True
        )
        return out, out_feats

    @graph_node
    def discriminator_mouth_dst(self) -> Tuple[Tensor, List[Tensor]]:
        out, out_feats = self.discriminator_mouth(
            self.mouth_roi_image_dst_normalized(), return_feats=True
        )
        return out, out_feats

    @graph_node
    def discriminate_left_eye_fake(self) -> Tensor:
        return self.discriminator_left_eye(
            self.left_eye_roi_image_generated().detach()
        )[0]

    @graph_node
    def discriminate_right_eye_fake(self) -> Tensor:
        return self.discriminator_right_eye(
            self.right_eye_roi_image_generated().detach()
        )[0]

    @graph_node
    def discriminate_mouth_fake(self) -> Tensor:
        return self.discriminator_mouth(self.mouth_roi_image_generated().detach())[0]

    @graph_node
    def discriminate_left_eye_dst(self) -> Tensor:
        return self.discriminator_left_eye(self.left_eye_roi_image_dst_normalized())[0]

    @graph_node
    def discriminate_right_eye_dst(self) -> Tensor:
        return self.discriminator_right_eye(self.right_eye_roi_image_dst_normalized())[
            0
        ]

    @graph_node
    def discriminate_mouth_dst(self) -> Tensor:
        return self.discriminator_mouth(self.mouth_roi_image_dst_normalized())[0]


class FaceGFPGANPipeline(GFPGANPipeline, FacePix2PixPipeline):
    def __init__(
        self,
        config: "FaceGFPGANConfig",
        generator_model: "nn.Module",
        discriminator_model: "StyleGAN2Discriminator",
        device: Union[str, torch.device],
        discriminator_left_eye: Optional["FacialComponentDiscriminator"] = None,
        discriminator_right_eye: Optional["FacialComponentDiscriminator"] = None,
        discriminator_mouth: Optional["FacialComponentDiscriminator"] = None,
    ):
        GFPGANPipeline.__init__(
            self,
            config=config,
            generator_model=generator_model,
            discriminator_model=discriminator_model,
            device=device,
        )
        FacePix2PixPipeline.__init__(
            self,
            config=config,
            device=device,
            discriminator_left_eye=discriminator_left_eye,
            discriminator_right_eye=discriminator_right_eye,
            discriminator_mouth=discriminator_mouth,
        )
        self.config: "FaceGFPGANConfig"


class FacePix2PixHDPipeline(Pix2PixHDPipeline, FacePix2PixPipeline):
    def __init__(
        self,
        config: "FacePix2PixHDConfig",
        generator_model: "nn.Module",
        discriminator_model: "MultiscaleDiscriminator",
        device: Union[str, torch.device],
        discriminator_left_eye: Optional["FacialComponentDiscriminator"] = None,
        discriminator_right_eye: Optional["FacialComponentDiscriminator"] = None,
        discriminator_mouth: Optional["FacialComponentDiscriminator"] = None,
    ):
        Pix2PixHDPipeline.__init__(
            self,
            config=config,
            generator_model=generator_model,
            discriminator_model=discriminator_model,
            device=device,
        )
        FacePix2PixPipeline.__init__(
            self,
            config=config,
            device=device,
            discriminator_left_eye=discriminator_left_eye,
            discriminator_right_eye=discriminator_right_eye,
            discriminator_mouth=discriminator_mouth,
        )
        self.config: "FacePix2PixHDConfig"


class FaceSpadePipeline(Pix2PixHDPipeline, FacePix2PixPipeline):
    def __init__(
        self,
        config: "FaceSpadeConfig",
        generator_model: "nn.Module",
        discriminator_model: "MultiscaleDiscriminator",
        device: Union[str, torch.device],
        discriminator_left_eye: Optional["FacialComponentDiscriminator"] = None,
        discriminator_right_eye: Optional["FacialComponentDiscriminator"] = None,
        discriminator_mouth: Optional["FacialComponentDiscriminator"] = None,
    ):
        Pix2PixHDPipeline.__init__(
            self,
            config=config,
            generator_model=generator_model,
            discriminator_model=discriminator_model,
            device=device,
        )
        FacePix2PixPipeline.__init__(
            self,
            config=config,
            device=device,
            discriminator_left_eye=discriminator_left_eye,
            discriminator_right_eye=discriminator_right_eye,
            discriminator_mouth=discriminator_mouth,
        )
        self.config: "FaceSpadeConfig"


def face_gfpgan_pipeline_from_config(
    config: "FaceGFPGANConfig",
    device: Union[str, torch.device],
) -> FaceGFPGANPipeline:
    return FaceGFPGANPipeline(
        config=config,
        generator_model=to_device(config.generator_model(), device=device),
        discriminator_model=to_device(config.discriminator_model(), device=device),
        device=device,
        discriminator_left_eye=to_device(
            config.discriminator_left_eye_model(), device=device
        ),
        discriminator_right_eye=to_device(
            config.discriminator_right_eye_model(), device=device
        ),
        discriminator_mouth=to_device(
            config.discriminator_mouth_model(), device=device
        ),
    )


def face_pix2pixhd_pipeline_from_config(
    config: "FacePix2PixHDConfig",
    device: Union[str, torch.device],
) -> FacePix2PixHDPipeline:
    return FacePix2PixHDPipeline(
        config=config,
        generator_model=to_device(config.generator_model(), device=device),
        discriminator_model=to_device(config.discriminator_model(), device=device),
        device=device,
        discriminator_left_eye=to_device(
            config.discriminator_left_eye_model(), device=device
        ),
        discriminator_right_eye=to_device(
            config.discriminator_right_eye_model(), device=device
        ),
        discriminator_mouth=to_device(
            config.discriminator_mouth_model(), device=device
        ),
    )


def face_spade_pipeline_from_config(
    config: "FaceSpadeConfig",
    device: Union[str, torch.device],
) -> FaceSpadePipeline:
    return FaceSpadePipeline(
        config=config,
        generator_model=to_device(config.generator_model(), device=device),
        discriminator_model=to_device(config.discriminator_model(), device=device),
        device=device,
        discriminator_left_eye=to_device(
            config.discriminator_left_eye_model(), device=device
        ),
        discriminator_right_eye=to_device(
            config.discriminator_right_eye_model(), device=device
        ),
        discriminator_mouth=to_device(
            config.discriminator_mouth_model(), device=device
        ),
    )
