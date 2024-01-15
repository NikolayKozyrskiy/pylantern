from typing import TYPE_CHECKING, List

from pylantern.inference.output_dispatcher import BaseInferenceOutputDispatcher

if TYPE_CHECKING:
    from pylantern.inference.face.deepfake.df_pipeline import DeepFakeInferencePipeline


class DeepFakeInferenceOutputDispatcher(BaseInferenceOutputDispatcher):
    def __init__(self, stage_names: List[str]) -> None:
        super().__init__(stage_names=stage_names)

    def swap(self, pipeline: "DeepFakeInferencePipeline", *args, **kwargs) -> None:
        pipeline.swap_face()

    def enhance_swapped_dst_img(
        self, pipeline: "DeepFakeInferencePipeline", *args, **kwargs
    ) -> None:
        pipeline.enhance_swapped_dst_img()

    def enhance_dst_img(
        self, pipeline: "DeepFakeInferencePipeline", *args, **kwargs
    ) -> None:
        pipeline.enhance_dst_img()

    def enhance_src_img(
        self, pipeline: "DeepFakeInferencePipeline", *args, **kwargs
    ) -> None:
        pipeline.enhance_src_img()
