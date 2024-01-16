from enum import Enum
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from pylantern.inference.pipeline import BaseInferencePipeline


class BaseInferenceOutputDispatcher:
    def __init__(self, stage_names: List[Enum]) -> None:
        self.stage_names = stage_names

    def compute_stages(
        self,
        pipeline: "BaseInferencePipeline",
        *args,
        **kwargs,
    ) -> None:
        for stage_name in self.stage_names:
            getattr(self, stage_name.value)(pipeline=pipeline, *args, **kwargs)
        return None
