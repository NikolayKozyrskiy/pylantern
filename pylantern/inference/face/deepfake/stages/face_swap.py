from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

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
        face_segmenter: Optional["FaceSegmenter"] = None,
        paste_back: bool = True,
    ) -> None:
        self.face_aligner = face_aligner
        self.face_swapper_model = face_swapper_model
        self.image_size = image_size
        self.predict_mask = predict_mask
        self.crop_paste_method = crop_paste_method
        self.paste_back = paste_back
        self.face_segmenter = face_segmenter

    @torch.no_grad()
    def swap_face(
        self,
        swap_data: "FaceSwapData",
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
        paste_back: bool = True,
    ) -> None:
        super().__init__(
            face_aligner=face_aligner,
            face_swapper_model=face_swapper_model,
            image_size=image_size,
            predict_mask=predict_mask,
            crop_paste_method=crop_paste_method,
            face_segmenter=face_segmenter,
            paste_back=paste_back,
        )

    @torch.no_grad()
    def swap_face(
        self,
        swap_data: "FaceSwapData",
        *args,
        **kwargs,
    ) -> "FaceSwapData":
        if swap_data.src_img is None and swap_data.src_face is None:
            raise ValueError("Either src_img or src_face must be non-none")
        if swap_data.dst_face is None:
            swap_data.dst_face = self.face_aligner.get_face(
                img=swap_data.dst_img, face_idx=swap_data.dst_face_idx
            )
        if swap_data.src_face is None:
            swap_data.src_face = self.face_aligner.get_face(
                img=swap_data.src_img, face_idx=swap_data.src_face_idx
            )
        if swap_data.dst_face is None or swap_data.src_face is None:
            swap_data.swapping_occurred = False
            return swap_data
        if self.predict_mask:
            (
                swap_data.swapped_dst_img,
                swap_data.transform_matrix,
            ) = self.face_swapper_model.get(
                img=swap_data.dst_img,
                target_face=swap_data.dst_face,
                source_face=swap_data.src_face,
                paste_back=False,
            )
            if self.paste_back:
                swap_data.predicted_dst_mask = self.face_segmenter.get_soft_mask(
                    img=cv2.resize(
                        swap_data.swapped_dst_img,
                        (320, 320),
                        interpolation=cv2.INTER_LANCZOS4,
                    )
                )
                swap_data.predicted_dst_mask = cv2.resize(
                    swap_data.predicted_dst_mask,
                    (128, 128),
                    interpolation=cv2.INTER_NEAREST,
                )
                swap_data.swapped_dst_img = self.face_aligner.paste_back_aligned(
                    face_swap_data=swap_data
                )
                swap_data.swapping_occurred = True
                return swap_data
        else:
            res = self.face_swapper_model.get(
                img=swap_data.dst_img,
                target_face=swap_data.dst_face,
                source_face=swap_data.src_face,
                paste_back=self.paste_back,
            )
            if self.paste_back:
                swap_data.swapped_dst_img = res
                swap_data.transform_matrix = None
            else:
                swap_data.swapped_dst_img = res[0]
                swap_data.transform_matrix = res[1]
            swap_data.swapping_occurred = True
        return swap_data


class FaceGeneratorSwapper(FaceSwapper):
    def __init__(
        self,
        face_aligner: "FaceAlignerInfa",
        face_swapper_model: "FaceGeneratorInferenceModel",
        device: Union[torch.device, str],
        image_size: Tuple[int, int],
        predict_mask: bool,
        crop_paste_method: "CropPasteMethod",
        paste_back: bool = True,
        face_segmenter: Optional["FaceSegmenter"] = None,
    ) -> None:
        super().__init__(
            face_aligner=face_aligner,
            face_swapper_model=face_swapper_model,
            image_size=image_size,
            predict_mask=predict_mask,
            crop_paste_method=crop_paste_method,
            face_segmenter=face_segmenter,
            paste_back=paste_back,
        )
        self.device = device
        self.face_swapper_model.to(device).eval()

    @torch.no_grad()
    def swap_face(
        self,
        swap_data: "FaceSwapData",
        *args,
        **kwargs,
    ) -> FaceSwapData:
        if isinstance(swap_data.dst_img, torch.Tensor):
            swap_data.dst_img = tensor_to_image(
                swap_data.dst_img, val_range=(0.0, 1.0), keepdim=False
            )
        if self.crop_paste_method == CropPasteMethod.BBOX:
            return self._swap_by_bbox_crop(
                swap_data=swap_data,
                *args,
                **kwargs,
            )
        elif self.crop_paste_method in [
            CropPasteMethod.INFA_INSWAPPER,
            CropPasteMethod.INFA_GENERATOR,
        ]:
            return self._swap_by_infa(
                swap_data=swap_data,
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
        swap_data: "FaceSwapData",
        *args,
        **kwargs,
    ) -> FaceSwapData:
        swap_data.dst_face_processed = self.face_aligner.get_face_processed(
            img=swap_data.dst_img, face_idx=swap_data.dst_face_idx
        )
        if swap_data.dst_face_processed.face is None:
            swap_data.swapping_occurred = False
            swap_data.dst_face_processed = None
            return swap_data

        swap_data.aligned_dst_img = self.face_aligner.get_crop_by_bbox_squared(
            face_processed=swap_data.dst_face_processed, crop_size=self.image_size
        )
        (
            predicted_dst_img,
            predicted_dst_mask,
        ) = self.face_swapper_model(
            image_to_tensor(bgr2rgb(swap_data.aligned_dst_img)).to(self.device)
        )
        swap_data.predicted_dst_img = rgb2bgr(tensor_to_image(predicted_dst_img))
        swap_data.predicted_dst_mask = (
            predicted_dst_mask.cpu()
            .numpy()
            .transpose(0, 2, 3, 1)
            .squeeze(0)
            .clip(0, 1.0)
        )
        swap_data.swapping_occurred = True
        if self.paste_back:
            swap_data.swapped_dst_img = self.face_aligner.paste_back_by_bbox_squared(
                face_swap_data=swap_data
            )
        return swap_data

    @torch.no_grad()
    def _swap_by_infa(
        self,
        swap_data: "FaceSwapData",
        *args,
        **kwargs,
    ) -> FaceSwapData:
        if swap_data.dst_face is None:
            swap_data.dst_face = self.face_aligner.get_face(
                img=swap_data.dst_img, face_idx=swap_data.dst_face_idx
            )
        if swap_data.dst_face is None:
            swap_data.swapping_occurred = False
            return swap_data
        (
            swap_data.aligned_dst_img,
            swap_data.transform_matrix,
        ) = self.face_aligner.get_one_aligned_img(
            img=swap_data.dst_img,
            face=swap_data.dst_face,
            face_idx=swap_data.dst_face_idx,
        )

        if self.crop_paste_method == CropPasteMethod.INFA_INSWAPPER:
            return self.__swap_by_infa_inswapper(swap_data=swap_data, *args, **kwargs)
        elif self.crop_paste_method == CropPasteMethod.INFA_GENERATOR:
            return self.__swap_by_predicted_mask(swap_data=swap_data, *args, **kwargs)
        else:
            raise ValueError(
                f"Given crop method {self.crop_paste_method} is not implemented for _swap_by_infa()"
            )

    @torch.no_grad()
    def __swap_by_infa_inswapper(
        self,
        swap_data: "FaceSwapData",
        *args,
        **kwargs,
    ) -> "FaceSwapData":
        if self.predict_mask:
            (
                predicted_dst_img,
                predicted_dst_mask,
            ) = self.face_swapper_model(
                image_to_tensor(bgr2rgb(swap_data.aligned_dst_img)).to(self.device)
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
            ) * swap_data.aligned_dst_img.astype(
                np.float32
            )
            swapped_dst_img = swapped_dst_img.clip(0, 255).round().astype(np.uint8)
        else:
            swapped_dst_img = self.face_swapper_model(
                image_to_tensor(bgr2rgb(swap_data.aligned_dst_img)).to(self.device)
            )
            swapped_dst_img = rgb2bgr(tensor_to_image(swapped_dst_img))

        swap_data.swapped_dst_img = swapped_dst_img
        swap_data.swapping_occurred = True
        if self.paste_back:
            swap_data.swapped_dst_img = self.face_aligner.paste_back_infa(
                face_swap_data=swap_data
            )
        return swap_data

    @torch.no_grad()
    def __swap_by_predicted_mask(
        self,
        swap_data: "FaceSwapData",
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
            image_to_tensor(bgr2rgb(swap_data.aligned_dst_img)).to(self.device)
        )
        predicted_dst_img = rgb2bgr(tensor_to_image(predicted_dst_img))
        predicted_dst_mask = (
            predicted_dst_mask.cpu()
            .numpy()
            .transpose(0, 2, 3, 1)
            .squeeze(0)
            .clip(0, 1.0)
        )
        swap_data.swapping_occurred = True
        swap_data.predicted_dst_img = predicted_dst_img
        swap_data.predicted_dst_mask = predicted_dst_mask
        if self.paste_back:
            swap_data.swapped_dst_img = self.face_aligner.paste_back_aligned(swap_data)
        else:
            swapped_dst_img = predicted_dst_img.astype(
                np.float32
            ) * predicted_dst_mask + (
                1 - predicted_dst_mask
            ) * swap_data.aligned_dst_img.astype(
                np.float32
            )
            swap_data.swapped_dst_img = (
                swapped_dst_img.clip(0, 255).round().astype(np.uint8)
            )
        return swap_data


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
