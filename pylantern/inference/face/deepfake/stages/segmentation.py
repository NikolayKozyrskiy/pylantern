from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
from facexlib.parsing import init_parsing_model
from facexlib.utils.misc import img2tensor
from torch import Tensor
from torch.nn import Module
from torchvision.transforms.functional import normalize

from pylantern.common.constants import DEFAULT_MEAN, DEFAULT_STD


class FaceSegmenter:
    def __init__(
        self,
        model: "Module",
        device: Union[str, torch.device],
    ) -> None:
        self.model = model
        self.device = device

    @torch.no_grad()
    def infer(
        self,
        img: np.ndarray,
        *args,
        **kwargs,
    ) -> np.ndarray:
        raise NotImplementedError

    @torch.no_grad()
    def get_mask(
        self,
        img: np.ndarray,
        *args,
        **kwargs,
    ) -> np.ndarray:
        raise NotImplementedError

    @torch.no_grad()
    def get_soft_mask(
        self,
        img: np.ndarray,
        *args,
        **kwargs,
    ) -> np.ndarray:
        raise NotImplementedError


class FaceSegmenterParsenet(FaceSegmenter):
    def __init__(
        self,
        model: "Module",
        device: Union[str, torch.device],
    ) -> None:
        super().__init__(model=model, device=device)
        self._input_shape = (512, 512)
        self._mask_colormap = [
            0,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            0,
            255,
            0,
            0,
            0,
        ]

    @torch.no_grad()
    def get_mask(
        self,
        img: np.ndarray,
        *args,
        **kwargs,
    ) -> np.ndarray:
        face_input = self._img2tensor(img)
        out = self._infer(face_input)
        mask = self._binarize(mask_pred=out, shape=img.shape[:2], soften=False)
        return mask

    @torch.no_grad()
    def get_soft_mask(
        self,
        img: np.ndarray,
        *args,
        **kwargs,
    ) -> np.ndarray:
        face_input = self._img2tensor(img)
        out = self._infer(face_input)
        mask = self._binarize(mask_pred=out, shape=img.shape[:2], soften=True)
        return mask

    @torch.no_grad()
    def _img2tensor(self, img: np.ndarray) -> Tensor:
        face_input = cv2.resize(img, self._input_shape, interpolation=cv2.INTER_LINEAR)
        face_input = img2tensor(
            face_input.astype("float32") / 255.0, bgr2rgb=True, float32=True
        )
        normalize(face_input, DEFAULT_MEAN, DEFAULT_STD, inplace=True)
        face_input = torch.unsqueeze(face_input, 0).to(self.device)
        return face_input

    @torch.no_grad()
    def _infer(self, img: Tensor) -> np.ndarray:
        mask = self.model(img)[0].argmax(dim=1)
        mask = mask.squeeze().cpu().numpy()
        return mask

    def _binarize(
        self,
        mask_pred: np.ndarray,
        shape: Tuple[int, int],
        soften: bool = True,
    ) -> np.ndarray:
        mask = np.zeros(mask_pred.shape)
        for idx, color in enumerate(self._mask_colormap):
            mask[mask_pred == idx] = color
        if soften:
            #  blur the mask
            mask = cv2.GaussianBlur(mask, (101, 101), 11)
            mask = cv2.GaussianBlur(mask, (101, 101), 11)
        # remove the black borders
        thres = 10
        mask[:thres, :] = 0
        mask[-thres:, :] = 0
        mask[:, :thres] = 0
        mask[:, -thres:] = 0
        mask = mask / 255.0

        mask = cv2.resize(mask, shape)
        return mask[:, :, None]


def load_face_segmenter_parsenet(
    weights_path: Optional[Path], device: Union[str, torch.device]
) -> "FaceSegmenterParsenet":
    if weights_path is None:
        weights_path = Path("_d/gfpgan/weights")
    model = init_parsing_model(
        model_name="parsenet", device=device, model_rootpath=weights_path
    ).to(device)
    return FaceSegmenterParsenet(model=model, device=device)


if __name__ == "__main__":
    pass
