from concurrent.futures import Executor
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from matches.shortcuts.optimizer import LRSchedulerWrapper, SchedulerScopeType
from torch.nn import Module
from torch.optim import Adam, Optimizer

from pylantern.common.constants import DEFAULT_IMG_MEAN, DEFAULT_IMG_STD
from pylantern.config import BaseConfig
from pylantern.tasks.gan.common.visualization.builder import PreviewImageConfig
from pylantern.tasks.gan.common.visualization.output import save_previews
from pylantern.tasks.gan.pix2pix.data.constants import DatasetType
from pylantern.tasks.gan.pix2pix.visualization.base import (
    original_image_dst_base,
    pred_image_base,
)

if TYPE_CHECKING:
    from pylantern.tasks.gan.pix2pix.pipelines import BasePix2PixPipeline


class BasePix2PixConfig(BaseConfig):
    predict_mask: bool = False

    dataset_type: DatasetType = DatasetType.IMAGE_MASK_FOLDERS
    attrs_to_load: Union[Sequence[str], Mapping[str, str]] = {
        "image": "imgs__infa_aligned.png",  # this is src image
        "image_dst": "imgs__inswapper__gfpganV14__sergey.png",
        # "mask_src": "masks__infa_aligned.png",
        # "mask": "masks__inswapper__gfpganV14__sergey.png",
    }
    split: str = "default"
    train_folds: Sequence[Any] = tuple(range(99))
    valid_folds: Sequence[Any] = (99,)

    image_size: Tuple[int, int] = (512, 512)
    img_ext: str = "png"

    beta1: float = 0.5
    beta2: float = 0.999

    preview_config: list[PreviewImageConfig] = [
        PreviewImageConfig(base=original_image_dst_base, overlays=[]),
        PreviewImageConfig(base=pred_image_base, overlays=[]),
    ]
    output_config: list[Callable[["BasePix2PixPipeline", Executor, Path], None]] = [
        save_previews()
    ]

    def generator_model(self, *args, **kwargs) -> "Module":
        raise NotImplementedError

    def discriminator_model(self, *args, **kwargs) -> "Module":
        raise NotImplementedError

    def optimizer_generator(self, model: Module) -> "Optimizer":
        return Adam(model.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

    def optimizer_discriminator(self, model: Module) -> "Optimizer":
        return Adam(model.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

    def scheduler_generator(self, optimizer: Optimizer) -> "LRSchedulerWrapper":
        return LRSchedulerWrapper(None, scope_type=SchedulerScopeType.BATCH)

    def scheduler_discriminator(self, optimizer: Optimizer) -> "LRSchedulerWrapper":
        return LRSchedulerWrapper(None, scope_type=SchedulerScopeType.BATCH)


if __name__ == "__main__":
    pass
