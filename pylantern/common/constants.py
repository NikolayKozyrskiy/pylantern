import os

import numpy as np

FFMPEG_BIN = (
    os.environ["FFMPEG_BIN"]
    if os.environ.get("FFMPEG_BIN", None) is not None
    else "ffmpeg"
)

ARCFACE_DST_KPS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)

IMG_8BIT_MAX_VAL = float(2**8 - 1)
IMG_16BIT_MAX_VAL = float(2**16 - 1)

DEFAULT_IMG_MEAN = np.array([0.5, 0.5, 0.5])
DEFAULT_IMG_STD = np.array([0.5, 0.5, 0.5])

IMAGENET_IMG_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_IMG_STD = np.array([0.229, 0.224, 0.225])

CLIP_IMG_MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
CLIP_IMG_STD = np.array([0.26862954, 0.26130258, 0.27577711])
