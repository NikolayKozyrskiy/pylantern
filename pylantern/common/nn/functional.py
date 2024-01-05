from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor


def mse_reduction(
    tensor_1: Union[Tensor, float],
    tensor_2: Union[Tensor, float],
    reduction: str = "mean",
    dim: Optional[List[int]] = None,
    keepdim: bool = False,
) -> Tensor:
    diff = (tensor_1 - tensor_2) ** 2
    return reduce_tensor(diff, reduction=reduction, dim=dim, keepdim=keepdim)


def l1_reduction(
    tensor_1: Union[Tensor, float],
    tensor_2: Union[Tensor, float],
    reduction: str = "mean",
    dim: Optional[List[int]] = None,
    keepdim: bool = False,
) -> Tensor:
    diff = abs(tensor_1 - tensor_2)
    return reduce_tensor(diff, reduction=reduction, dim=dim, keepdim=keepdim)


def reduce_tensor(
    input_tensor: Tensor,
    reduction: str,
    dim: Optional[List[int]] = None,
    keepdim: bool = False,
) -> Tensor:
    if reduction == "mean":
        return input_tensor.mean(dim=dim, keepdim=keepdim)
    elif reduction == "sum":
        return input_tensor.sum(dim=dim, keepdim=keepdim)
    return input_tensor


# TODO: vectorize loop
def umeyama(
    src_points: Tensor, dst_points: Tensor, estimate_scale: bool = True
) -> Tensor:
    r"""
    Estimate the transformation from a set of corresponding points.

    You can determine the over-, well- and under-determined parameters
    with the total least-squares method.

    Number of source and destination coordinates must match.

    Parameters
    ----------
    src_points : (B, N, 2) torch.Tensor
        Source coordinates.
    dst_points : (B, N, 2) torch.Tensor
        Destination coordinates.

    Returns
    ----------
    M : (B, 2, 3) torch.Tensor
        The homogeneous similarity transformation matrices. The matrices contain
        NaN values only if the problem is not well-conditioned.

    References
    ----------
    [1] "Least-squares estimation of transformation parameters between two
        point patterns", Shinji Umeyama, PAMI 1991, :DOI:`10.1109/34.88573`
    """
    device = src_points.device
    batch_size = src_points.shape[0]

    def _run_umeyama(src: Tensor, dst: Tensor):
        num = src.shape[0]
        dim = src.shape[1]

        # Compute mean of src and dst.
        src_mean = src.mean(dim=0)
        dst_mean = dst.mean(dim=0)

        # Subtract mean from src and dst.
        src_demean = src - src_mean
        dst_demean = dst - dst_mean

        # Eq. (38).
        A = dst_demean.T @ src_demean / num

        # Eq. (39).
        d = torch.ones(dim, dtype=torch.float64, device=device)
        if torch.linalg.det(A) < 0:
            d[dim - 1] = -1

        T = torch.eye(dim + 1, dtype=torch.float64, device=device)

        U, S, V = torch.linalg.svd(A)
        V = V.T

        # Eq. (40) and (43).
        rank = torch.linalg.matrix_rank(A)
        if rank == 0:
            return torch.nan * T
        elif rank == dim - 1:
            if torch.linalg.det(U) * torch.linalg.det(V) > 0:
                T[:dim, :dim] = U @ V
            else:
                s = d[dim - 1]
                d[dim - 1] = -1
                T[:dim, :dim] = U @ torch.diag(d) @ V
                d[dim - 1] = s
        else:
            T[:dim, :dim] = U @ torch.diag(d) @ V

        if estimate_scale:
            # Eq. (41) and (42).
            scale = 1.0 / src_demean.var(dim=0, unbiased=False).sum() * (S @ d)
        else:
            scale = 1.0

        T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean)
        T[:dim, :dim] *= scale

        return T[:2]

    transform_matrices = torch.zeros((batch_size, 2, 3), device=device)
    for i in range(batch_size):
        transform_matrices[i] = _run_umeyama(src_points[i], dst_points[i])

    return transform_matrices


def atm_arcface(src_points: Tensor, dst_points: Tensor, image_size: int) -> Tensor:
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    dst = dst_points.expand_as(src_points) * ratio
    dst[..., 0] += diff_x
    return umeyama(src_points, dst, estimate_scale=True)


def atm_by_lstsq(src_points: Tensor, dst_points: Tensor) -> Tensor:
    transform_matrix = torch.as_tensor(
        [[1, 0, 0], [0, 1, 0]], dtype=torch.float64, device=src_points.device
    )
    num_points = src_points.shape[0]
    src_pts_ = torch.hstack(
        [
            src_points,
            torch.ones(
                (num_points, 1), dtype=src_points.dtype, device=src_points.device
            ),
        ]
    )
    dst_pts_ = torch.hstack(
        [
            dst_points,
            torch.ones(
                (num_points, 1), dtype=src_points.dtype, device=src_points.device
            ),
        ]
    )

    A, res, rank, s = torch.linalg.lstsq(src_pts_, dst_pts_)
    if rank == 3:
        transform_matrix = torch.as_tensor(
            [[A[0, 0], A[1, 0], A[2, 0]], [A[0, 1], A[1, 1], A[2, 1]]],
            dtype=A.dtype,
            device=src_points.device,
        )
    elif rank == 2:
        transform_matrix = torch.as_tensor(
            [[A[0, 0], A[1, 0], 0], [A[0, 1], A[1, 1], 0]],
            dtype=A.dtype,
            device=src_points.device,
        )
    return transform_matrix


def gram_matrix(x: Tensor) -> Tensor:
    """Calculate Gram matrix.

    Args:
        x (torch.Tensor): Tensor with shape of (n, c, h, w).

    Returns:
        torch.Tensor: Gram matrix.
    """
    n, c, h, w = x.size()
    features = x.view(n, c, w * h)
    features_t = features.transpose(1, 2)
    return features.bmm(features_t) / (c * h * w)


def fspecial_gauss_1d(
    size: int,
    sigma: float,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Create 1-D gauss kernel.

    Args:
        size: the size of gauss kernel.
        sigma: sigma of normal distribution.

    Returns:
        1D kernel (size).
    """
    coords = torch.arange(size, device=device, dtype=dtype)
    coords -= size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    return g.reshape(-1)


def fspecial_gauss_2d(
    size: int,
    sigma: float,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Create 2-D gauss kernel.

    Args:
        size: the size of gauss kernel.
        sigma: sigma of normal distribution.

    Returns:
        2D kernel (size x size).
    """
    gaussian_vec = fspecial_gauss_1d(size, sigma, device, dtype)
    return torch.outer(gaussian_vec, gaussian_vec)


def rgb2gray(img: Tensor, keepdim: bool = True) -> Tensor:
    out_gray = (
        0.2989 * img[:, 0, :, :] + 0.5870 * img[:, 1, :, :] + 0.1140 * img[:, 2, :, :]
    )
    if keepdim:
        out_gray = out_gray.unsqueeze(1)
    return out_gray
