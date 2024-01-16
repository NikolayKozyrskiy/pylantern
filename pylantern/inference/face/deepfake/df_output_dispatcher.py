from enum import Enum
from typing import TYPE_CHECKING, List

from pylantern.inference.output_dispatcher import BaseInferenceOutputDispatcher

if TYPE_CHECKING:
    from pylantern.inference.face.deepfake.df_pipeline import DeepFakeInferencePipeline


class DeepFakeInferenceOutputDispatcher(BaseInferenceOutputDispatcher):
    def __init__(self, stage_names: List[Enum]) -> None:
        super().__init__(stage_names=stage_names)

    def swap(self, pipeline: "DeepFakeInferencePipeline", *args, **kwargs) -> None:
        pipeline.swap_face()

    def enhance_swapped_dst(
        self, pipeline: "DeepFakeInferencePipeline", *args, **kwargs
    ) -> None:
        pipeline.enhance_swapped_dst_img()

    def enhance_dst(
        self, pipeline: "DeepFakeInferencePipeline", *args, **kwargs
    ) -> None:
        pipeline.enhance_dst_img()

    def enhance_src(
        self, pipeline: "DeepFakeInferencePipeline", *args, **kwargs
    ) -> None:
        pipeline.enhance_src_img()


class DeepFakeStageNames(str, Enum):
    SWAP = DeepFakeInferenceOutputDispatcher.swap.__name__
    ENHANCE_SWAPPED_DST = DeepFakeInferenceOutputDispatcher.enhance_swapped_dst.__name__
    ENHANCE_DST = DeepFakeInferenceOutputDispatcher.enhance_dst.__name__
    ENHANCE_SRC = DeepFakeInferenceOutputDispatcher.enhance_src.__name__
