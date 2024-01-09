from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

from matches.callbacks import Callback, TqdmProgressCallback
from matches.loop import Loop

from pylantern.config import BaseModel

if TYPE_CHECKING:
    from pylantern.inference.pipeline import BaseInferencePipeline


class BaseInferenceConfig(BaseModel):
    root_path: Path = Path("_d")

    transforms: list[Callable] = []
    workers_num: int = 4
    checkpoint_path: Optional[Path] = None

    def resume(self, loop: Loop, pipeline: "BaseInferencePipeline") -> None:
        if self.checkpoint_path is not None:
            loop.state_manager.read_state(
                self.checkpoint_path,
                skip_keys=[
                    "scheduler",
                ],
            )

    def preprocess(self, loop: Loop, pipeline: "BaseInferencePipeline") -> None:
        pass

    def postprocess(self, loop: Loop, pipeline: "BaseInferencePipeline") -> None:
        pass

    def callbacks(self, *args, **kwargs) -> List["Callback"]:
        return [TqdmProgressCallback()]


if __name__ == "__main__":
    pass
