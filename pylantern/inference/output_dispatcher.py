from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from pylantern.inference.pipeline import BaseInferencePipeline


class BaseInferenceOutputDispatcher:
    def __init__(self, stage_names: List[str]) -> None:
        self.stage_names = stage_names

    def compute_stages(
        self,
        pipeline: "BaseInferencePipeline",
        *args,
        **kwargs,
    ) -> None:
        for stage_name in self.stage_names:
            getattr(self, stage_name)(pipeline, *args, **kwargs)
        return None
