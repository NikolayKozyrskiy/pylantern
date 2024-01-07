from typing import List, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torchvision import models

from pylantern.common.constants import IMAGENET_IMG_MEAN, IMAGENET_IMG_STD
from pylantern.common.nn.functional import gram_matrix


class Vgg19(nn.Module):
    def __init__(self, requires_grad: bool = False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X: Tensor) -> List[Tensor]:
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGG19Loss(nn.Module):
    def __init__(
        self,
        style_weight: float = 0.0,
        normalize_inputs: bool = True,
        resize_inputs: bool = False,
        reduction: str = "mean",
    ):
        super(VGG19Loss, self).__init__()
        self.style_weight = style_weight
        self.normalize_inputs = normalize_inputs
        self.resize_inputs = resize_inputs

        self.vgg = Vgg19().eval()
        self.criterion = nn.L1Loss(reduction=reduction)
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.register_buffer(
            "mean",
            torch.tensor(IMAGENET_IMG_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_IMG_STD, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self._vgg_img_size = (224, 224)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.normalize_inputs:
            input = (input - self.mean) / self.std
            target = (target - self.mean) / self.std
        if self.resize_inputs:
            input = F.interpolate(
                input, mode="bilinear", size=self._vgg_img_size, align_corners=False
            )
            target = F.interpolate(
                target, mode="bilinear", size=self._vgg_img_size, align_corners=False
            )

        input_features, target_features = self.vgg(input), self.vgg(target)
        loss = 0.0
        for i in range(len(input_features)):
            loss += self.weights[i] * self.criterion(
                input_features[i], target_features[i]
            )
        if self.style_weight > 0.0:
            for i in range(len(input_features)):
                loss += (
                    self.style_weight
                    * self.weights[i]
                    * self.criterion(
                        gram_matrix(input_features[i]), gram_matrix(target_features[i])
                    )
                )
        return loss
