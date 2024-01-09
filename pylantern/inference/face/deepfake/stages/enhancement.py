import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import cv2
import numpy as np
import torch
from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from torch.nn import Module
from torch.nn import functional as F
from torchvision.transforms.functional import normalize

from pylantern.common.utils import get_device
from pylantern.common.utils.img import bgr2rgb, rgb2bgr

# from align_faces import get_reference_facial_points, warp_and_crop_face
# from sr_model.real_esrnet import RealESRNet
# from .detection import RetinaFaceDetection
# from .parsing import FaceParse
# from ..modules.gpen import FaceGAN

if TYPE_CHECKING:
    from facexlib.utils.face_restoration_helper import FaceRestoreHelper

    from pylantern.inference.face.deepfake.df_config import DeepFakeInferenceConfig


def denoise_nlmeans(
    img: np.ndarray,
    h: int = 2,
    hColor: int = 3,
    templateWindowSize: int = 3,
    searchWindowSize: int = 21,
) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(
        img,
        None,
        h=h,
        hColor=hColor,
        templateWindowSize=templateWindowSize,
        searchWindowSize=searchWindowSize,
    )


@dataclass
class FaceEnhancerResult:
    restored_img: Optional[np.ndarray] = None
    cropped_faces: Optional[List[np.ndarray]] = None
    restored_faces: Optional[List[np.ndarray]] = None


class FaceEnhancer:
    def __init__(
        self,
        enhancement_model: "Module",
        device: Union[str, torch.device],
    ) -> None:
        self.enhancement_model = enhancement_model
        self.device = device

    @torch.no_grad()
    def enhance(
        self,
        img: np.ndarray,
        is_aligned: bool = False,
        only_keep_largest: bool = True,
        paste_back: bool = True,
        *args,
        **kwargs,
    ) -> FaceEnhancerResult:
        raise NotImplementedError


class FaceEnhancerTencent(FaceEnhancer):
    def __init__(
        self,
        enhancement_model: "Module",
        face_helper: "FaceRestoreHelper",
        device: Union[str, torch.device],
    ) -> None:
        super().__init__(enhancement_model=enhancement_model, device=device)
        self.face_helper: "FaceRestoreHelper" = face_helper

    @torch.no_grad()
    def enhance(
        self,
        img: np.ndarray,
        is_aligned: bool = False,
        only_keep_largest: bool = True,
        paste_back: bool = True,
        weight: float = 0.5,
        *args,
        **kwargs,
    ) -> "FaceEnhancerResult":
        self.face_helper.clean_all()

        if is_aligned:  # the inputs are already aligned
            img = cv2.resize(img, (512, 512))
            self.face_helper.cropped_faces = [img]
        else:
            self.face_helper.read_image(img)
            self.face_helper.get_face_landmarks_5(
                only_keep_largest=only_keep_largest, eye_dist_threshold=5
            )
            self.face_helper.align_warp_face()

        # face restoration
        for cropped_face in self.face_helper.cropped_faces:
            cropped_face_t = img2tensor(
                cropped_face / 255.0, bgr2rgb=True, float32=True
            )
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

            try:
                output = self.enhancement_model(
                    cropped_face_t, return_rgb=False, weight=weight
                )[0]
                restored_face = tensor2img(
                    output.squeeze(0), rgb2bgr=True, min_max=(-1, 1)
                )
            except RuntimeError as error:
                print(f"\tFailed inference for GFPGAN: {error}.")
                restored_face = cropped_face

            restored_face = restored_face.astype("uint8")
            self.face_helper.add_restored_face(restored_face)

        if not is_aligned and paste_back:
            self.face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            restored_img = self.face_helper.paste_faces_to_input_image(
                upsample_img=None
            )
            return FaceEnhancerResult(
                cropped_faces=self.face_helper.cropped_faces,
                restored_faces=self.face_helper.restored_faces,
                restored_img=restored_img,
            )

        else:
            return FaceEnhancerResult(
                cropped_faces=self.face_helper.cropped_faces,
                restored_faces=self.face_helper.restored_faces,
                restored_img=None,
            )


def load_face_enhancer_tencent(
    enhancement_model: Optional["Module"],
    face_helper: "FaceRestoreHelper",
    device: Union[str, torch.device, None] = None,
) -> "FaceEnhancerTencent":
    if device is None:
        device = get_device()
    enhancement_model = enhancement_model.to(device).eval()
    return FaceEnhancerTencent(
        enhancement_model=enhancement_model,
        face_helper=face_helper,
        device=device,
    )


class FaceEnhancerCodeformer(FaceEnhancer):
    def __init__(
        self,
        enhancement_model: Optional["Module"],
        face_helper: "FaceRestoreHelper",
        device: Union[str, torch.device],
    ) -> None:
        super().__init__(enhancement_model=enhancement_model, device=device)
        self.face_helper: "FaceRestoreHelper" = face_helper

    @torch.no_grad()
    def enhance(
        self,
        img: np.ndarray,
        is_aligned: bool = False,
        only_keep_largest: bool = True,
        paste_back: bool = True,
        weight: float = 1.0,
        adain: bool = True,
        *args,
        **kwargs,
    ) -> "FaceEnhancerResult":
        self.face_helper.clean_all()

        if is_aligned:  # the inputs are already aligned
            img = cv2.resize(img, (512, 512))
            self.face_helper.cropped_faces = [img]
        else:
            self.face_helper.read_image(img)
            self.face_helper.get_face_landmarks_5(
                only_keep_largest=only_keep_largest, eye_dist_threshold=5
            )
            self.face_helper.align_warp_face()

        # face restoration
        for cropped_face in self.face_helper.cropped_faces:
            cropped_face_t = img2tensor(
                cropped_face / 255.0, bgr2rgb=True, float32=True
            )
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

            try:
                output = self.enhancement_model(cropped_face_t, w=weight, adain=adain)[
                    0
                ]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            except RuntimeError as error:
                print(f"\tFailed inference for CodeFormer: {error}.")
                restored_face = cropped_face

            restored_face = restored_face.astype("uint8")
            self.face_helper.add_restored_face(restored_face)

        if not is_aligned and paste_back:
            self.face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            restored_img = self.face_helper.paste_faces_to_input_image(
                upsample_img=None
            )
            return FaceEnhancerResult(
                cropped_faces=self.face_helper.cropped_faces,
                restored_faces=self.face_helper.restored_faces,
                restored_img=restored_img,
            )

        else:
            return FaceEnhancerResult(
                cropped_faces=self.face_helper.cropped_faces,
                restored_faces=self.face_helper.restored_faces,
                restored_img=None,
            )


def load_face_enhancer_codeformer(
    enhancement_model: Optional["Module"],
    face_helper: "FaceRestoreHelper",
    device: Union[str, torch.device, None] = None,
) -> Optional["FaceEnhancerCodeformer"]:
    if device is None:
        device = get_device()
    enhancement_model = enhancement_model.to(device).eval()
    return FaceEnhancerCodeformer(
        enhancement_model=enhancement_model,
        face_helper=face_helper,
        device=device,
    )


class EnhancerRealESRGAN(FaceEnhancer):
    """A helper class for upsampling images with RealESRGAN.

    Args:
        scale (int): Upsampling scale factor used in the networks. It is usually 2 or 4.
        tile (int): As too large images result in the out of GPU memory issue, so this tile option will first crop
            input images into tiles, and then process each of them. Finally, they will be merged into one image.
            0 denotes for do not use tile. Default: 0.
        tile_pad (int): The pad size for each tile, to remove border artifacts. Default: 10.
        pre_pad (int): Pad the input images to avoid border artifacts. Default: 10.
    """

    def __init__(
        self,
        enhancement_model: "Module",
        device: Union[str, torch.device],
        scale: int,
        tile: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 10,
        half_precision: bool = False,
    ) -> None:
        super().__init__(enhancement_model=enhancement_model, device=device)
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.half_precision = half_precision
        self.enhancement_model = enhancement_model
        self.mod_scale = None

    @torch.no_grad()
    def enhance(
        self,
        img: np.ndarray,
        *args,
        **kwargs,
    ) -> "FaceEnhancerResult":
        if len(img.shape) == 2:  # gray image
            img_mode = "L"
            _img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_mode = "RGB"
            _img = bgr2rgb(img)

        max_range = 65535 if np.max(_img) > 256 else 255
        _img = _img / max_range

        # ------------------- process image ------------------- #
        self.pre_process(_img)
        if self.tile_size > 0:
            self.tile_process()
        else:
            self.process()
        output_img = self.post_process(downsample_to_orig=True)
        output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
        if img_mode == "L":
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

        # ------------------------------ return ------------------------------ #
        if max_range == 65535:  # 16-bit image
            output = (output_img * 65535.0).round().astype(np.uint16)
        else:
            output = (output_img * 255.0).round().astype(np.uint8)

        return FaceEnhancerResult(restored_img=output)

    def pre_process(self, img: np.ndarray) -> None:
        """Pre-process, such as pre-pad and mod pad, so that the images can be divisible"""
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        self.img = img.unsqueeze(0).to(self.device)
        if self.half_precision:
            self.img = self.img.half()

        # pre_pad
        if self.pre_pad != 0:
            self.img = F.pad(self.img, (0, self.pre_pad, 0, self.pre_pad), "reflect")
        # mod pad for divisible borders
        if self.scale == 2:
            self.mod_scale = 2
        elif self.scale == 1:
            self.mod_scale = 4
        if self.mod_scale is not None:
            self.mod_pad_h, self.mod_pad_w = 0, 0
            _, _, h, w = self.img.size()
            if h % self.mod_scale != 0:
                self.mod_pad_h = self.mod_scale - h % self.mod_scale
            if w % self.mod_scale != 0:
                self.mod_pad_w = self.mod_scale - w % self.mod_scale
            self.img = F.pad(
                self.img, (0, self.mod_pad_w, 0, self.mod_pad_h), "reflect"
            )
        return None

    def process(self) -> None:
        # enhancement_model inference
        self.output = self.enhancement_model(self.img)

    def tile_process(self) -> None:
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.

        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[
                    :,
                    :,
                    input_start_y_pad:input_end_y_pad,
                    input_start_x_pad:input_end_x_pad,
                ]

                # upscale tile
                try:
                    with torch.no_grad():
                        output_tile = self.enhancement_model(input_tile)
                except RuntimeError as error:
                    print("Error", error)

                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                self.output[
                    :, :, output_start_y:output_end_y, output_start_x:output_end_x
                ] = output_tile[
                    :,
                    :,
                    output_start_y_tile:output_end_y_tile,
                    output_start_x_tile:output_end_x_tile,
                ]

    def post_process(self, downsample_to_orig: bool = False) -> torch.Tensor:
        # remove extra pad
        if self.mod_scale is not None:
            _, _, h, w = self.output.size()
            self.output = self.output[
                :,
                :,
                0 : h - self.mod_pad_h * self.scale,
                0 : w - self.mod_pad_w * self.scale,
            ]
        # remove prepad
        if self.pre_pad != 0:
            _, _, h, w = self.output.size()
            self.output = self.output[
                :,
                :,
                0 : h - self.pre_pad * self.scale,
                0 : w - self.pre_pad * self.scale,
            ]
        if downsample_to_orig:
            self.output = F.interpolate(
                self.output, scale_factor=1 / self.scale, mode="nearest"
            )
        return self.output


def load_enhancer_realesrgan(
    enhancement_model: "Module",
    scale: int,
    device: Optional[str] = None,
    tile: int = 0,
    tile_pad: int = 10,
    pre_pad: int = 10,
    half_precision: bool = False,
) -> "EnhancerRealESRGAN":
    if device is None:
        device = get_device()
    enhancement_model = enhancement_model.to(device).eval()
    if half_precision:
        enhancement_model.half()
    return EnhancerRealESRGAN(
        enhancement_model=enhancement_model,
        device=device,
        scale=scale,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half_precision=half_precision,
    )


# class FaceEnhancement(object):
#     def __init__(
#         self,
#         args,
#         base_dir="./",
#         in_size=512,
#         out_size=None,
#         enhancement_model=None,
#         use_sr=True,
#         device="cuda",
#     ):
#         self.facedetector = RetinaFaceDetection(base_dir, device)
#         self.facegan = FaceGAN(
#             base_dir,
#             in_size,
#             out_size,
#             enhancement_model,
#             args.channel_multiplier,
#             args.narrow,
#             args.key,
#             device=device,
#         )
#         self.srmodel = RealESRNet(
#             base_dir, args.sr_model, args.sr_scale, args.tile_size, device=device
#         )
#         self.faceparser = FaceParse(base_dir, device=device)
#         self.use_sr = use_sr
#         self.in_size = in_size
#         self.out_size = in_size if out_size is None else out_size
#         self.threshold = 0.9
#         self.alpha = args.alpha

#         # the mask for pasting restored faces back
#         self.mask = np.zeros((512, 512), np.float32)
#         cv2.rectangle(self.mask, (26, 26), (486, 486), (1, 1, 1), -1, cv2.LINE_AA)
#         self.mask = cv2.GaussianBlur(self.mask, (101, 101), 4)
#         self.mask = cv2.GaussianBlur(self.mask, (101, 101), 4)

#         self.kernel = np.array(
#             ([0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]),
#             dtype="float32",
#         )

#         # get the reference 5 landmarks position in the crop settings
#         default_square = True
#         inner_padding_factor = 0.25
#         outer_padding = (0, 0)
#         self.reference_5pts = get_reference_facial_points(
#             (self.in_size, self.in_size),
#             inner_padding_factor,
#             outer_padding,
#             default_square,
#         )

#     def mask_postprocess(self, mask, thres=26):
#         mask[:thres, :] = 0
#         mask[-thres:, :] = 0
#         mask[:, :thres] = 0
#         mask[:, -thres:] = 0
#         mask = cv2.GaussianBlur(mask, (101, 101), 4)
#         mask = cv2.GaussianBlur(mask, (101, 101), 4)
#         return mask.astype(np.float32)

#     def process(self, img, aligned=False):
#         orig_faces, enhanced_faces = [], []
#         if aligned:
#             ef = self.facegan.process(img)
#             orig_faces.append(img)
#             enhanced_faces.append(ef)

#             if self.use_sr:
#                 ef = self.srmodel.process(ef)

#             return ef, orig_faces, enhanced_faces

#         if self.use_sr:
#             img_sr = self.srmodel.process(img)
#             if img_sr is not None:
#                 img = cv2.resize(img, img_sr.shape[:2][::-1])

#         facebs, landms = self.facedetector.detect(img)

#         height, width = img.shape[:2]
#         full_mask = np.zeros((height, width), dtype=np.float32)
#         full_img = np.zeros(img.shape, dtype=np.uint8)

#         for i, (faceb, facial5points) in enumerate(zip(facebs, landms)):
#             if faceb[4] < self.threshold:
#                 continue
#             fh, fw = (faceb[3] - faceb[1]), (faceb[2] - faceb[0])

#             facial5points = np.reshape(facial5points, (2, 5))

#             of, tfm_inv = warp_and_crop_face(
#                 img,
#                 facial5points,
#                 reference_pts=self.reference_5pts,
#                 crop_size=(self.in_size, self.in_size),
#             )

#             # enhance the face
#             ef = self.facegan.process(of)

#             orig_faces.append(of)
#             enhanced_faces.append(ef)

#             # tmp_mask = self.mask
#             tmp_mask = self.mask_postprocess(self.faceparser.process(ef)[0] / 255.0)
#             tmp_mask = cv2.resize(tmp_mask, (self.in_size, self.in_size))
#             tmp_mask = cv2.warpAffine(tmp_mask, tfm_inv, (width, height), flags=3)

#             if min(fh, fw) < 100:  # gaussian filter for small faces
#                 ef = cv2.filter2D(ef, -1, self.kernel)

#             ef = cv2.addWeighted(ef, self.alpha, of, 1.0 - self.alpha, 0.0)

#             if self.in_size != self.out_size:
#                 ef = cv2.resize(ef, (self.in_size, self.in_size))
#             tmp_img = cv2.warpAffine(ef, tfm_inv, (width, height), flags=3)

#             mask = tmp_mask - full_mask
#             full_mask[np.where(mask > 0)] = tmp_mask[np.where(mask > 0)]
#             full_img[np.where(mask > 0)] = tmp_img[np.where(mask > 0)]

#         full_mask = full_mask[:, :, np.newaxis]
#         if self.use_sr and img_sr is not None:
#             img = cv2.convertScaleAbs(img_sr * (1 - full_mask) + full_img * full_mask)
#         else:
#             img = cv2.convertScaleAbs(img * (1 - full_mask) + full_img * full_mask)

#         return img, orig_faces, enhanced_faces
