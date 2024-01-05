from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nn.functional import fspecial_gauss_2d

# Based on kornia


class MSDSSIMLoss(nn.Module):
    r"""Creates a criterion that computes MS-DSSIM:
        multi-scale structural dissimilarity loss

    Reference:
        [1]: https://research.nvidia.com/sites/default/files/pubs/2017-03_Loss-Functions-for/NN_ImgProc.pdf#page11

    Args:
        sigmas: gaussian sigma values.
        data_range: the range of the images.
        K: k values.
        reduction : Specifies the reduction to apply to the
         output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
         ``'mean'``: the sum of the output will be divided by the number of elements
         in the output, ``'sum'``: the output will be summed.

    Returns:
        The computed loss.

    Shape:
        - Input1: :math:`(N, C, H, W)`.
        - Input2: :math:`(N, C, H, W)`.
        - Output: :math:`(N, H, W)` or scalar if reduction is set to ``'mean'`` or ``'sum'``.

    Examples:
        >>> pred = torch.rand(1, 3, 5, 5)
        >>> gt = torch.rand(1, 3, 5, 5)
        >>> criterion = MSDSSIMLoss()
        >>> loss = criterion(pred, gt)
    """

    def __init__(
        self,
        sigmas: List[float] = [0.5, 1.0, 2.0, 4.0, 8.0],
        data_range: float = 1.0,
        K: Tuple[float, float] = (0.01, 0.03),
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.DR: float = data_range
        self.C1: float = (K[0] * data_range) ** 2
        self.C2: float = (K[1] * data_range) ** 2
        self.pad = int(2 * sigmas[-1])
        self.reduction: str = reduction

        filter_size = int(4 * sigmas[-1] + 1)
        g_masks = torch.zeros((3 * len(sigmas), 1, filter_size, filter_size))

        # Compute mask at different scales
        for idx, sigma in enumerate(sigmas):
            g_masks[3 * idx + 0, 0, :, :] = fspecial_gauss_2d(filter_size, sigma)
            g_masks[3 * idx + 1, 0, :, :] = fspecial_gauss_2d(filter_size, sigma)
            g_masks[3 * idx + 2, 0, :, :] = fspecial_gauss_2d(filter_size, sigma)

        self.register_buffer("_g_masks", g_masks)

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute MS-SSIM loss.

        Args:
            img1: the predicted image with shape :math:`(B, C, H, W)`.
            img2: the target image with a shape of :math:`(B, C, H, W)`.

        Returns:
            Estimated MS-SSIM loss.
        """
        if not isinstance(predicted, torch.Tensor):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(predicted)}")

        if not isinstance(target, torch.Tensor):
            raise TypeError(f"Output type is not a torch.Tensor. Got {type(target)}")

        if not len(predicted.shape) == len(target.shape):
            raise ValueError(
                f"Input shapes should be same. Got {type(predicted)} and {type(target)}."
            )

        g_masks: torch.Tensor = torch.jit.annotate(torch.Tensor, self._g_masks)

        CH: int = predicted.shape[-3]
        mux = F.conv2d(predicted, g_masks, groups=CH, padding=self.pad)
        muy = F.conv2d(target, g_masks, groups=CH, padding=self.pad)
        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = (
            F.conv2d(predicted * predicted, g_masks, groups=CH, padding=self.pad) - mux2
        )
        sigmay2 = F.conv2d(target * target, g_masks, groups=CH, padding=self.pad) - muy2
        sigmaxy = (
            F.conv2d(predicted * target, g_masks, groups=CH, padding=self.pad) - muxy
        )

        lc = (2 * muxy + self.C1) / (mux2 + muy2 + self.C1)
        cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)
        lM = lc[:, -1, :, :] * lc[:, -2, :, :] * lc[:, -3, :, :]
        PIcs = cs.prod(dim=1)

        loss = 1 - lM * PIcs

        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)

        return loss
