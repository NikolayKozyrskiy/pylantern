from typing import Any, List, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from pylantern.tasks.gan.pix2pix.models.generators.generator import (
    GeneratorInferenceModel,
    GeneratorModel,
)


class FaceGeneratorModel(GeneratorModel):
    def __init__(
        self,
        generator_model: nn.Module,
        mean: Sequence[float],
        std: Sequence[float],
        input_range: Sequence[int],
    ) -> None:
        super().__init__(
            generator_model=generator_model, mean=mean, std=std, input_range=input_range
        )

    def predict_image(self, input: Tensor, *args, **kwargs) -> Union[Tensor, List[Any]]:
        return self.preprocessor.destandardize(
            self.generator_model.predict(
                self.preprocessor.predict(input), *args, **kwargs
            )[:, :3, ...]
        )

    def predict_image_and_mask(
        self, input: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        output = self.generator_model.predict(
            self.preprocessor.predict(input), *args, **kwargs
        )
        img = self.preprocessor.destandardize(output[:, :3, ...])
        mask = torch.sigmoid(output[:, 3:, ...])
        return img, mask


class FaceGeneratorInferenceModel(GeneratorInferenceModel):
    def __init__(
        self, generator_model: nn.Module, mean: Sequence[float], std: Sequence[float]
    ) -> None:
        super().__init__(generator_model=generator_model, mean=mean, std=std)

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        output = self.generator_model(self.preprocessor(input))
        img = self.preprocessor.destandardize(output[:, :3, ...])
        mask = torch.sigmoid(output[:, 3:, ...])
        return img, mask
