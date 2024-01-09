from contextlib import contextmanager
from typing import TYPE_CHECKING, Union

import torch

from pylantern.inference.pipeline import BaseInferencePipeline

if TYPE_CHECKING:
    from pylantern.inference.face.deepfake.data import FaceSwapData
    from pylantern.inference.face.deepfake.df_config import DeepFakeInferenceConfig


class DeepFakeInferencePipeline(BaseInferencePipeline):
    def __init__(
        self,
        config: "DeepFakeInferenceConfig",
        device: Union[str, torch.device, None] = None,
    ):
        BaseInferencePipeline.__init__(self, config=config, device=device)
        self.face_aligner = config.face_aligner()
        self.face_swapper = config.face_swapper(
            face_aligner=self.face_aligner, device=device
        )
        self.face_enhancer = config.face_enhancer(device=device)

    @contextmanager
    def data_scope(self, data: "FaceSwapData"):
        try:
            self.data = data
            yield
        finally:
            self.data = None
