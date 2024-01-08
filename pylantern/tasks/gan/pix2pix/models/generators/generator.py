from typing import Any, List, Sequence, Union

from torch import Tensor, nn

from pylantern.tasks.gan.pix2pix.models.general.preprocessor import (
    Preprocessor,
    PreprocessorInferenceWrapper,
)


class GeneratorModel(nn.Module):
    def __init__(
        self,
        generator_model: nn.Module,
        mean: Sequence[float],
        std: Sequence[float],
        input_range: Sequence[int],
    ) -> None:
        super().__init__()
        self.generator_model = generator_model
        self.preprocessor = Preprocessor(mean=mean, std=std, input_range=input_range)

    @property
    def mean(self):
        return self.preprocessor.mean

    @property
    def std(self):
        return self.preprocessor.std

    @property
    def input_range(self):
        return self.preprocessor.input_range

    def forward(self, input: Tensor, *args, **kwargs) -> Union[Tensor, List[Any]]:
        return self.generator_model(self.preprocessor(input), *args, **kwargs)


class GeneratorInferenceModel(nn.Module):
    def __init__(
        self, generator_model: nn.Module, mean: Sequence[float], std: Sequence[float]
    ) -> None:
        super().__init__()
        self.generator_model = generator_model
        self.preprocessor = PreprocessorInferenceWrapper(mean=mean, std=std)

    def forward(self, input: Tensor, *args, **kwargs) -> Union[Tensor, List[Any]]:
        return self.generator_model(self.preprocessor(input), *args, **kwargs)
