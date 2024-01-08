from typing import Optional, Sequence

import torch
from torch import Tensor, nn


class Preprocessor(nn.Module):
    def __init__(
        self,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        input_range: Optional[Sequence[int]] = None,
        **kwargs
    ):
        super().__init__()
        if mean is not None:
            self.register_buffer(
                "mean", torch.tensor(mean).view(3, 1, 1).float(), persistent=False
            )
        if std is not None:
            self.register_buffer(
                "std", torch.tensor(std).view(3, 1, 1).float(), persistent=False
            )

        if input_range is not None:
            self.register_buffer(
                "input_range", torch.tensor(input_range).float(), persistent=False
            )
            if input_range[1] == 1:
                self.register_buffer(
                    "div", torch.tensor(255.0).float(), persistent=False
                )

    def forward(self, x: Tensor) -> Tensor:
        out = x
        if out.dtype != torch.float32:
            out = out.float()

        if hasattr(self, "div") and hasattr(self, "mean") and hasattr(self, "std"):
            out = (out / self.div - self.mean.expand_as(x)) / self.std.expand_as(x)

        return out

    def predict(self, x: Tensor) -> Tensor:
        if x.dtype != torch.float32:
            x = x.float()

        if hasattr(self, "div"):
            x /= self.div

        if hasattr(self, "mean"):
            x -= self.mean.expand_as(x)

        if hasattr(self, "std"):
            x /= self.std.expand_as(x)

        return x

    def destandardize(self, x: Tensor) -> Tensor:
        if hasattr(self, "std"):
            x *= self.std.expand_as(x)

        if hasattr(self, "mean"):
            x += self.mean.expand_as(x)

        if hasattr(self, "input_range"):
            torch.clip_(x, min=0.0, max=self.input_range[1])

        return x


class PreprocessorInferenceWrapper(nn.Module):
    def __init__(
        self,
        mean: Sequence[float],
        std: Sequence[float],
    ):
        super().__init__()
        self.register_buffer(
            "mean", torch.tensor(mean).reshape(3, 1, 1).float(), persistent=True
        )
        self.register_buffer(
            "std", torch.tensor(std).reshape(3, 1, 1).float(), persistent=True
        )
        self.register_buffer("div", torch.tensor(255.0).float(), persistent=True)

    def forward(self, x: Tensor) -> Tensor:
        out = (x / self.div - self.mean) / self.std
        return out

    def destandardize(self, x: Tensor) -> Tensor:
        x = x * self.std + self.mean
        torch.clip_(x, min=0.0, max=1.0)
        return x
