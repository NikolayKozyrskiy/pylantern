from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from insightface.app.common import Face

from pylantern.common.utils import get_device
from pylantern.common.utils.img import (
    bgr2rgb,
    image_to_tensor,
    rgb2bgr,
    tensor_to_image,
)
from pylantern.inference.face.deepfake.data import CropPasteMethod, FaceSwapData

if TYPE_CHECKING:
    from insightface.model_zoo.inswapper import INSwapper

    from pylantern.inference.face.deepfake.stages.alignment import FaceAlignerInfa
    from pylantern.inference.face.deepfake.stages.segmentation import FaceSegmenter
    from pylantern.tasks.gan.pix2pix.models import FaceGeneratorInferenceModel


class FaceSwapper:
    def __init__(
        self,
        face_aligner: "FaceAlignerInfa",
        face_swapper_model: Union["FaceGeneratorInferenceModel", "INSwapper"],
        image_size: Tuple[int, int],
        predict_mask: bool,
        crop_paste_method: "CropPasteMethod",
    ) -> None:
        self.face_aligner = face_aligner
        self.face_swapper_model = face_swapper_model
        self.image_size = image_size
        self.predict_mask = predict_mask
        self.crop_paste_method = crop_paste_method

    @torch.no_grad()
    def swap_face(
        self,
        dst_img: np.ndarray,
        src_img: Optional[np.ndarray] = None,
        dst_face: Optional["Face"] = None,
        src_face: Optional["Face"] = None,
        src_face_idx: Optional[int] = None,
        dst_face_idx: Optional[int] = None,
        crop_by_bbox: bool = False,
        paste_back: bool = True,
        *args,
        **kwargs,
    ) -> FaceSwapData:
        raise NotImplementedError


class FaceSwapperINFA(FaceSwapper):
    def __init__(
        self,
        face_aligner: "FaceAlignerInfa",
        face_swapper_model: "INSwapper",
        image_size: Tuple[int, int],
        predict_mask: bool,
        crop_paste_method: "CropPasteMethod",
        face_segmenter: Optional["FaceSegmenter"] = None,
    ) -> None:
        super().__init__(
            face_aligner=face_aligner,
            face_swapper_model=face_swapper_model,
            image_size=image_size,
            predict_mask=predict_mask,
            crop_paste_method=crop_paste_method,
        )
        self.face_segmenter = face_segmenter

    @torch.no_grad()
    def swap_face(
        self,
        dst_img: np.ndarray,
        src_img: Optional[np.ndarray] = None,
        dst_face: Optional[Face] = None,
        src_face: Optional[Face] = None,
        src_face_idx: Optional[int] = None,
        dst_face_idx: Optional[int] = None,
        crop_by_bbox: bool = False,
        paste_back: bool = True,
        *args,
        **kwargs,
    ) -> "FaceSwapData":
        if src_img is None and src_face is None:
            raise ValueError("Either src_img or src_face must be non-none")
        if dst_face is None:
            dst_face = self.face_aligner.get_face(dst_img, face_idx=dst_face_idx)
        if src_face is None:
            src_face = self.face_aligner.get_face(src_img, face_idx=src_face_idx)
        if dst_face is None or src_face is None:
            return FaceSwapData(
                src_img=src_img,
                dst_img=dst_img,
                src_face_idx=src_face_idx,
                dst_face_idx=dst_face_idx,
                swapped_dst_img=dst_img,
                swapping_occurred=False,
            )
        if self.predict_mask:
            swapped_dst_img, transform_matrix = self.face_swapper_model.get(
                dst_img, dst_face, src_face, paste_back=False
            )
            if paste_back:
                predicted_dst_mask = self.face_segmenter.get_soft_mask(
                    cv2.resize(
                        swapped_dst_img, (320, 320), interpolation=cv2.INTER_LANCZOS4
                    )
                )
                predicted_dst_mask = cv2.resize(
                    predicted_dst_mask, (128, 128), interpolation=cv2.INTER_NEAREST
                )
                face_data = FaceSwapData(
                    src_img=src_img,
                    dst_img=dst_img,
                    src_face_idx=src_face_idx,
                    dst_face_idx=dst_face_idx,
                    transform_matrix=transform_matrix,
                    predicted_dst_img=swapped_dst_img,
                    predicted_dst_mask=predicted_dst_mask,
                    swapping_occurred=True,
                )
                face_data.swapped_dst_img = self.face_aligner.paste_back_aligned(
                    face_data
                )
                return face_data
        else:
            if paste_back:
                swapped_dst_img = self.face_swapper_model.get(
                    dst_img, dst_face, src_face, paste_back=paste_back
                )
                transform_matrix = None
            else:
                swapped_dst_img, transform_matrix = self.face_swapper_model.get(
                    dst_img, dst_face, src_face, paste_back=paste_back
                )
        return FaceSwapData(
            src_img=src_img,
            dst_img=dst_img,
            src_face=src_face,
            dst_face=dst_face,
            src_face_idx=src_face_idx,
            dst_face_idx=dst_face_idx,
            transform_matrix=transform_matrix,
            swapped_dst_img=swapped_dst_img,
            swapping_occurred=True,
        )


class FaceGeneratorSwapper(FaceSwapper):
    def __init__(
        self,
        face_aligner: "FaceAlignerInfa",
        face_swapper_model: "FaceGeneratorInferenceModel",
        device: Union[torch.device, str],
        image_size: Tuple[int, int],
        predict_mask: bool,
        crop_paste_method: "CropPasteMethod",
    ) -> None:
        super().__init__(
            face_aligner=face_aligner,
            face_swapper_model=face_swapper_model,
            image_size=image_size,
            predict_mask=predict_mask,
            crop_paste_method=crop_paste_method,
        )
        self.device = device
        self.face_swapper_model.to(device).eval()

    @torch.no_grad()
    def swap_face(
        self,
        dst_img: Union[np.ndarray, torch.Tensor],
        src_img: Union[np.ndarray, torch.Tensor, None] = None,
        dst_face: Optional["Face"] = None,
        src_face: Optional["Face"] = None,
        src_face_idx: Optional[int] = None,
        dst_face_idx: Optional[int] = None,
        paste_back: bool = True,
        *args,
        **kwargs,
    ) -> FaceSwapData:
        if isinstance(dst_img, torch.Tensor):
            dst_img = tensor_to_image(dst_img, val_range=(0.0, 1.0), keepdim=False)
        if self.crop_paste_method == CropPasteMethod.BBOX:
            return self._swap_by_bbox_crop(
                dst_img=dst_img,
                src_img=src_img,
                dst_face=dst_face,
                src_face=src_face,
                src_face_idx=src_face_idx,
                dst_face_idx=dst_face_idx,
                paste_back=paste_back,
                *args,
                **kwargs,
            )
        elif self.crop_paste_method in [
            CropPasteMethod.INFA_INSWAPPER,
            CropPasteMethod.INFA_GENERATOR,
        ]:
            return self._swap_by_infa(
                dst_img=dst_img,
                src_img=src_img,
                dst_face=dst_face,
                src_face=src_face,
                src_face_idx=src_face_idx,
                dst_face_idx=dst_face_idx,
                paste_back=paste_back,
                *args,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Given crop method {self.crop_paste_method} is not implemented for FaceGeneratorSwapper"
            )

    @torch.no_grad()
    def _swap_by_bbox_crop(
        self,
        dst_img: np.ndarray,
        src_img: Union[np.ndarray, torch.Tensor, None] = None,
        dst_face: Optional[Face] = None,
        src_face: Optional[Face] = None,
        src_face_idx: Optional[int] = None,
        dst_face_idx: Optional[int] = None,
        paste_back: bool = True,
        *args,
        **kwargs,
    ) -> FaceSwapData:
        dst_face_processed = self.face_aligner.get_face_processed(
            dst_img, face_idx=dst_face_idx
        )
        if dst_face_processed.face is None:
            return FaceSwapData(
                dst_img=dst_img,
                dst_face_idx=dst_face_idx,
                dst_face_processed=None,
                swapped_dst_img=dst_img,
                swapping_occurred=False,
            )
        cropped_img = self.face_aligner.get_crop_by_bbox_squared(
            face_processed=dst_face_processed, crop_size=self.image_size
        )
        (
            predicted_dst_img,
            predicted_dst_mask,
        ) = self.face_swapper_model(
            image_to_tensor(bgr2rgb(cropped_img)).to(self.device)
        )
        predicted_dst_img = rgb2bgr(tensor_to_image(predicted_dst_img))
        predicted_dst_mask = (
            predicted_dst_mask.cpu()
            .numpy()
            .transpose(0, 2, 3, 1)
            .squeeze(0)
            .clip(0, 1.0)
        )
        face_data = FaceSwapData(
            dst_img=dst_img,
            dst_face=dst_face_processed.face,
            dst_face_idx=dst_face_idx,
            aligned_dst_img=cropped_img,
            dst_face_processed=dst_face_processed,
            predicted_dst_img=predicted_dst_img,
            predicted_dst_mask=predicted_dst_mask,
            swapped_dst_img=predicted_dst_img,
            swapping_occurred=True,
        )
        if paste_back:
            face_data.swapped_dst_img = self.face_aligner.paste_back_by_bbox_squared(
                face_data
            )
        return face_data

    @torch.no_grad()
    def _swap_by_infa(
        self,
        dst_img: np.ndarray,
        src_img: Union[np.ndarray, torch.Tensor, None] = None,
        dst_face: Optional[Face] = None,
        src_face: Optional[Face] = None,
        src_face_idx: Optional[int] = None,
        dst_face_idx: Optional[int] = None,
        paste_back: bool = True,
        *args,
        **kwargs,
    ) -> FaceSwapData:
        if dst_face is None:
            dst_face = self.face_aligner.get_face(dst_img, face_idx=dst_face_idx)
        if dst_face is None:
            return FaceSwapData(
                dst_img=dst_img,
                dst_face_idx=dst_face_idx,
                swapped_dst_img=dst_img,
                swapping_occurred=False,
            )
        aligned_dst_img, transform_matrix = self.face_aligner.get_one_aligned_img(
            img=dst_img, face=dst_face, face_idx=dst_face_idx
        )
        face_data = FaceSwapData(
            dst_img=dst_img,
            dst_face=dst_face,
            dst_face_idx=dst_face_idx,
            aligned_dst_img=aligned_dst_img,
            transform_matrix=transform_matrix,
        )
        if self.crop_paste_method == CropPasteMethod.INFA_INSWAPPER:
            return self.__swap_by_infa_inswapper(
                face_data=face_data, paste_back=paste_back, *args, **kwargs
            )
        elif self.crop_paste_method == CropPasteMethod.INFA_GENERATOR:
            return self.__swap_by_predicted_mask(
                face_data=face_data, paste_back=paste_back, *args, **kwargs
            )
        else:
            raise ValueError(
                f"Given crop method {self.crop_paste_method} is not implemented for _swap_by_infa()"
            )

    @torch.no_grad()
    def __swap_by_infa_inswapper(
        self,
        face_data: "FaceSwapData",
        paste_back: bool = True,
        *args,
        **kwargs,
    ) -> "FaceSwapData":
        if self.predict_mask:
            (
                predicted_dst_img,
                predicted_dst_mask,
            ) = self.face_swapper_model(
                image_to_tensor(bgr2rgb(face_data.aligned_dst_img)).to(self.device)
            )
            predicted_dst_img = rgb2bgr(tensor_to_image(predicted_dst_img))
            predicted_dst_mask = (
                predicted_dst_mask.cpu()
                .numpy()
                .transpose(0, 2, 3, 1)
                .squeeze(0)
                .clip(0, 1.0)
            )
            swapped_dst_img = predicted_dst_img.astype(
                np.float32
            ) * predicted_dst_mask + (
                1 - predicted_dst_mask
            ) * face_data.aligned_dst_img.astype(
                np.float32
            )
            swapped_dst_img = swapped_dst_img.clip(0, 255).round().astype(np.uint8)
        else:
            swapped_dst_img = self.face_swapper_model(
                image_to_tensor(bgr2rgb(face_data.aligned_dst_img)).to(self.device)
            )
            swapped_dst_img = rgb2bgr(tensor_to_image(swapped_dst_img))

        face_data.swapped_dst_img = swapped_dst_img
        face_data.swapping_occurred = True
        if paste_back:
            face_data.swapped_dst_img = self.face_aligner.paste_back_infa(face_data)
        return face_data

    @torch.no_grad()
    def __swap_by_predicted_mask(
        self,
        face_data: "FaceSwapData",
        paste_back: bool = True,
        *args,
        **kwargs,
    ) -> "FaceSwapData":
        assert (
            self.predict_mask
        ), "predicted mask is mandatory for __swap_by_predicted_mask()"
        (
            predicted_dst_img,
            predicted_dst_mask,
        ) = self.face_swapper_model(
            image_to_tensor(bgr2rgb(face_data.aligned_dst_img)).to(self.device)
        )
        predicted_dst_img = rgb2bgr(tensor_to_image(predicted_dst_img))
        predicted_dst_mask = (
            predicted_dst_mask.cpu()
            .numpy()
            .transpose(0, 2, 3, 1)
            .squeeze(0)
            .clip(0, 1.0)
        )
        face_data.swapping_occurred = True
        face_data.predicted_dst_img = predicted_dst_img
        face_data.predicted_dst_mask = predicted_dst_mask
        if paste_back:
            face_data.swapped_dst_img = self.face_aligner.paste_back_aligned(face_data)
        else:
            swapped_dst_img = predicted_dst_img.astype(
                np.float32
            ) * predicted_dst_mask + (
                1 - predicted_dst_mask
            ) * face_data.aligned_dst_img.astype(
                np.float32
            )
            face_data.swapped_dst_img = (
                swapped_dst_img.clip(0, 255).round().astype(np.uint8)
            )
        return face_data


def load_face_swapper_infa(
    face_swapper_model: "INSwapper",
    face_aligner: "FaceAlignerInfa",
    image_size: Tuple[int, int],
    predict_mask: bool,
    crop_paste_method: "CropPasteMethod",
    face_segmenter: Optional["FaceSegmenter"] = None,
) -> "FaceSwapperINFA":
    if predict_mask:
        assert (
            face_segmenter is not None
        ), "face_segmenter must be initialized if predict_mask is True"
    return FaceSwapperINFA(
        face_aligner=face_aligner,
        face_swapper_model=face_swapper_model,
        face_segmenter=face_segmenter,
        image_size=image_size,
        predict_mask=predict_mask,
        crop_paste_method=crop_paste_method,
    )


def load_face_swapper_generator(
    face_swapper_model: "FaceGeneratorInferenceModel",
    face_aligner: "FaceAlignerInfa",
    image_size: Tuple[int, int],
    predict_mask: bool,
    crop_paste_method: "CropPasteMethod",
    device: Optional[str] = None,
) -> "FaceGeneratorSwapper":
    if device is None:
        device = get_device()
    return FaceGeneratorSwapper(
        face_aligner=face_aligner,
        face_swapper_model=face_swapper_model,
        device=device,
        image_size=image_size,
        predict_mask=predict_mask,
        crop_paste_method=crop_paste_method,
    )
