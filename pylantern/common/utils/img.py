import random
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import kornia
import numpy as np
from torch import Tensor
from torch.nn import functional as F

DEFAULT_MEAN = np.array([0.5, 0.5, 0.5])
DEFAULT_STD = np.array([0.5, 0.5, 0.5])

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711])


def load_img(src_path: Path, convert_bgr2rgb: bool = False) -> Optional[np.ndarray]:
    try:
        img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
        if convert_bgr2rgb:
            img = bgr2rgb(img)
        return img
    except FileNotFoundError:
        print(f"Could not load image by path: {src_path}")
        return None


def save_img(img: np.ndarray, dst_path: Path, convert_rgb2bgr: bool = False) -> None:
    if convert_rgb2bgr:
        img = rgb2bgr(img)
    cv2.imwrite(str(dst_path), img)
    return None


def bgr2rgb(img: np.ndarray) -> np.ndarray:
    if img.dtype in [np.float32, np.float64, float]:
        img = img.astype(int)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb2bgr(img: np.ndarray) -> np.ndarray:
    if img.dtype in [np.float32, np.float64, float]:
        img = img.astype(int)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def choose_interpolation(
    img: np.ndarray,
    dsize: Tuple[int, int],
    downscale_interpolation: int = cv2.INTER_AREA,
    upscale_interpolation: int = cv2.INTER_LANCZOS4,
) -> int:
    if dsize[0] < img.shape[1] and dsize[1] < img.shape[0]:
        interpolation = downscale_interpolation
    else:
        interpolation = upscale_interpolation
    return interpolation


def crop_by_bbox(
    img: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]],
    crop_size: Union[Tuple[int, int], None] = None,
    interpolation: Optional[int] = None,
) -> Optional[np.ndarray]:
    if bbox is not None:
        x_l, y_l, x_r, y_r = bbox
        cropped_img = img[y_l:y_r, x_l:x_r, :].copy()
        if crop_size is not None:
            interpolation = (
                choose_interpolation(img=img, dsize=crop_size)
                if interpolation is None
                else interpolation
            )
            cropped_img = cv2.resize(
                cropped_img, dsize=crop_size, interpolation=interpolation
            )
        return cropped_img
    return None


def interpolate_img_tensor(
    img: Tensor,
    size: Union[Tuple[int, int], int],
    mode: str = "bilinear",
    align_corners: bool = False,
) -> Tensor:
    img = F.interpolate(
        img,
        size,
        mode=mode,
        align_corners=align_corners,
    )
    return img


def crop_random_params(
    h_orig: int,
    w_orig: int,
    h_crop_len: Optional[int] = None,
    w_crop_len: Optional[int] = None,
    crop_ratio: Optional[float] = None,
) -> Tuple[int, int, int, int]:
    if all(v is None for v in (h_crop_len, w_crop_len, crop_ratio)):
        raise ValueError("Either crop lens or crop ratio must be defined")
    if h_crop_len is None:
        h_crop_len = int(round(h_orig * crop_ratio))
    if w_crop_len is None:
        w_crop_len = int(round(w_orig * crop_ratio))
    h_crop_start = random.randint(0, h_orig - h_crop_len)
    w_crop_start = random.randint(0, w_orig - w_crop_len)
    return h_crop_start, h_crop_len, w_crop_start, w_crop_len


def crop_img_tensor_by_params(
    img: Tensor,
    h_crop_start: Tensor,
    h_crop_len: Tensor,
    w_crop_start: Tensor,
    w_crop_len: Tensor,
) -> Tensor:
    img_cropped = img[
        ...,
        w_crop_start : w_crop_start + w_crop_len,
        h_crop_start : h_crop_start + h_crop_len,
    ]
    return img_cropped


def image_to_tensor(
    image: np.ndarray,
    val_range: Optional[Tuple[float, float]] = None,
    keepdim: bool = False,
) -> Tensor:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Input type is not a np.ndarray. Got {type(image)}")
    img_tensor = kornia.image_to_tensor(image, keepdim=keepdim).float()
    if val_range is not None:
        img_tensor = (img_tensor - val_range[0]) / (val_range[1] - val_range[0])
    return img_tensor


def tensor_to_image(
    tensor: Tensor,
    val_range: Tuple[float, float] = (0.0, 1.0),
    keepdim: bool = False,
) -> "np.ndarray":
    if not isinstance(tensor, Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(tensor)}")

    if len(tensor.shape) > 4 or len(tensor.shape) < 2:
        raise ValueError("Input size must be a two, three or four dimensional tensor")

    input_shape = tensor.shape
    image: "np.ndarray" = tensor.cpu().detach().numpy()

    if len(input_shape) == 2:
        # (H, W) -> (H, W)
        pass
    elif len(input_shape) == 3:
        # (C, H, W) -> (H, W, C)
        if input_shape[0] == 1:
            # Grayscale for proper plt.imshow needs to be (H,W)
            image = image.squeeze()
        else:
            image = image.transpose(1, 2, 0)
    elif len(input_shape) == 4:
        # (B, C, H, W) -> (B, H, W, C)
        image = image.transpose(0, 2, 3, 1)
        if input_shape[0] == 1 and not keepdim:
            image = image.squeeze(0)
        if input_shape[1] == 1:
            image = image.squeeze(-1)
    else:
        raise ValueError(f"Cannot process tensor with shape {input_shape}")

    if val_range[0] == -1.0 and val_range[1] == 1.0:
        image = image / 2.0 + 0.5

    image = (image * 255).clip(0, 255).round().astype(np.uint8)
    return image
