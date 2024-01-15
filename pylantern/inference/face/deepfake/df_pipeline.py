from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch
from matches.shortcuts.dag import graph_node

from pylantern.common.utils import get_device
from pylantern.common.utils.img import load_img
from pylantern.inference.face.deepfake.stages.enhancement import denoise_nlmeans
from pylantern.inference.pipeline import BaseInferencePipeline

if TYPE_CHECKING:
    from pathlib import PurePath

    from insightface.app.common import Face

    from pylantern.inference.face.deepfake.data import FaceSwapData
    from pylantern.inference.face.deepfake.df_config import DeepFakeInferenceConfig


class DeepFakeInferencePipeline(BaseInferencePipeline):
    def __init__(
        self,
        config: "DeepFakeInferenceConfig",
        device: Union[str, torch.device, None] = None,
    ):
        BaseInferencePipeline.__init__(self, config=config, device=device)
        self.config: "DeepFakeInferenceConfig"
        self.face_aligner = config.face_aligner()
        self.face_swapper = config.face_swapper(
            face_aligner=self.face_aligner, device=device
        )
        self.face_enhancer = config.face_enhancer(device=device)
        self.src_img: Optional["np.ndarray"] = None
        self.src_face: Optional["Face"] = None

    @contextmanager
    def src_img_scope(
        self,
        src_img: Union["np.ndarray", "PurePath", str],
        face_idx: Optional[int] = None,
    ):
        if self.src_img is not None:
            raise ValueError("Entering src_img_scope() twice is redundant")
        try:
            self.src_img = (
                src_img if isinstance(src_img, np.ndarray) else load_img(src_img)
            )
            self.src_face = self.face_aligner.get_face(
                img=self.src_img, face_idx=face_idx
            )
            yield
        finally:
            self.src_img = None
            self.src_face = None

    @contextmanager
    def data_item_scope(self, data: "FaceSwapData"):
        try:
            self.data = data
            if self.src_img is not None and self.src_face is not None:
                self.data.src_img = self.src_img
                self.data.src_face = self.src_face
            yield
        finally:
            self.data = None

    @graph_node
    def compute_src_face(self) -> None:
        self.data.src_face = self.src_face
        return None

    @graph_node
    def compute_dst_face(self) -> None:
        self.data.dst_face = self.face_aligner.get_face(
            img=self.data.dst_img, face_idx=self.data.dst_face_idx
        )
        return None

    @graph_node
    def swap_face(self) -> None:
        self.data = self.face_swapper.swap_face(swap_data=self.data)
        return None

    @graph_node
    def enhance_swapped_dst_img(self) -> None:
        if self.data.swapping_occurred:
            self.data.enhanced_swapped_dst_img = self.face_enhancer.enhance(
                img=self.data.swapped_dst_img,
                is_aligned=False,
                only_keep_largest=True,
                paste_back=self.config.paste_back,
            ).restored_img
            if (
                self.config.denoise_enhanced
                and self.data.enhanced_swapped_dst_img is not None
            ):
                self.data.enhanced_swapped_dst_img = denoise_nlmeans(
                    self.data.enhanced_swapped_dst_img
                )
        return None

    @graph_node
    def enhance_dst_img(self) -> None:
        self.data.dst_img = self.face_enhancer.enhance(
            img=self.data.dst_img,
            is_aligned=False,
            only_keep_largest=True,
            paste_back=self.config.paste_back,
        ).restored_img
        return None

    @graph_node
    def enhance_src_img(self) -> None:
        self.src_img = self.face_enhancer.enhance(
            img=self.src_img,
            is_aligned=False,
            only_keep_largest=True,
            paste_back=self.config.paste_back,
        ).restored_img
        return None

    @graph_node
    def get_swapped_dst_img_soft_mask(self) -> Optional["np.ndarray"]:
        if self.face_swapper.face_segmenter is not None:
            return self.face_swapper.face_segmenter.get_soft_mask(
                img=self.data.swapped_dst_img
            )

    @graph_node
    def get_swapped_dst_img_binary_mask(self) -> Optional["np.ndarray"]:
        if self.face_swapper.face_segmenter is not None:
            return self.face_swapper.face_segmenter.get_mask(
                img=self.data.swapped_dst_img
            )

    @graph_node
    def get_dst_img_soft_mask(self) -> Optional["np.ndarray"]:
        if self.face_swapper.face_segmenter is not None:
            return self.face_swapper.face_segmenter.get_soft_mask(img=self.data.dst_img)

    @graph_node
    def get_dst_img_binary_mask(self) -> Optional["np.ndarray"]:
        if self.face_swapper.face_segmenter is not None:
            return self.face_swapper.face_segmenter.get_mask(img=self.data.dst_img)

    @graph_node
    def get_src_img_soft_mask(self) -> Optional["np.ndarray"]:
        if self.face_swapper.face_segmenter is not None:
            return self.face_swapper.face_segmenter.get_soft_mask(img=self.src_img)

    @graph_node
    def get_src_img_binary_mask(self) -> Optional["np.ndarray"]:
        if self.face_swapper.face_segmenter is not None:
            return self.face_swapper.face_segmenter.get_mask(img=self.src_img)


def deepfake_inference_pipeline_from_config(
    config: "DeepFakeInferenceConfig",
    device: Union[str, torch.device, None] = None,
) -> "DeepFakeInferencePipeline":
    return DeepFakeInferencePipeline(config=config, device=device)
