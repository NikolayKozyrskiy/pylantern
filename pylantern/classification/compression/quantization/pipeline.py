import contextlib
from typing import List, Optional, NamedTuple, TYPE_CHECKING

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Module
from matches.shortcuts.dag import graph_node

from ...pipeline import ClassificationPipeline
from .config import QuantClassificationConfig

if TYPE_CHECKING:
    from ...data.dataset import ClassificationDatasetItem


class QuantClassificationPipeline(ClassificationPipeline):
    def __init__(
        self,
        config: "QuantClassificationConfig",
        model: Module,
    ):
        super().__init__(config, model)
        self.config: QuantClassificationConfig = config
        self.do_quantize_fc = config.do_quantize_fc

        self.saved_params = []
        self.target_modules = []
        self.params_num = None

    @torch.no_grad()
    def save_params(self):
        if self.params_num is None:
            self.params_num = 0
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d) or (
                    isinstance(m, nn.Linear) and self.do_quantize_fc
                ):
                    self.params_num += 1
                    self.saved_params.append(m.weight.data.clone())
                    self.target_modules.append(m.weight)
        else:
            for idx in range(self.params_num):
                self.saved_params[idx].copy_(self.target_modules[idx].data)
        return None

    @torch.no_grad()
    def restore_weights(self) -> None:
        if self.config.num_bits_w >= 32:
            return None
        for idx in range(self.params_num):
            self.target_modules[idx].data.copy_(self.saved_params[idx])
        return None

    @torch.no_grad()
    def quantize_weights(self) -> None:
        if self.config.num_bits_w >= 32:
            return None
        self.save_params()
        bound = 2 ** (self.config.num_bits_w - 1) - 1
        for idx in range(self.params_num):
            w = self.target_modules[idx].data
            if len(w.size()) == 4:
                scales = (
                    torch.max(torch.max(torch.max(torch.abs(w), 3)[0], 2)[0], 1)[0]
                    / bound
                )
                scales = torch.reshape(scales, (-1, 1, 1, 1))
            elif len(w.size()) == 2:
                scales = torch.max(torch.abs(w), 1)[0] / bound
                scales = torch.reshape(scales, (-1, 1))
            else:
                raise ValueError(f"Unsupported w.size(): {w.size()}")
            w = w / scales
            w = torch.round(w)
            w = torch.clamp(w, -bound, bound)
            w = w * scales
            self.target_modules[idx].data = w
        return None


def pipeline_from_config(
    config: "QuantClassificationConfig", device: str
) -> QuantClassificationPipeline:
    model = config.model().to(device)
    return QuantClassificationPipeline(config, model=model)
