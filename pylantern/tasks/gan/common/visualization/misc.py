from enum import Enum

import torch


class Colors(Enum):
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    VIOLET = (255, 0, 255)
    YELLOW = (255, 255, 0)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)


@torch.no_grad()
def preview_points(
    image: torch.Tensor,
    points: torch.Tensor,
    color: Colors = Colors.WHITE,
    point_size: int = 3,
):
    n_images, c, h, w = image.shape
    points = points.round().long()
    n_images_p, n_points, n_coords = points.shape

    color = (
        torch.as_tensor(color.value, device=image.device, dtype=image.dtype)
        .expand(n_images, n_points, 3)
        .contiguous()
    )

    r = (point_size - 1) // 2
    pts = points.new_tensor([(i, j) for i in range(-r, r) for j in range(-r, r)]).view(
        1, -1, 1, 2
    )
    points = (points.unsqueeze(1) + pts).view(n_images, -1, 2)
    color = color.unsqueeze(1).expand(-1, pts.shape[1], -1, 3).reshape(n_images, -1, 2)

    is_in_image_bounds = (
        (points < points.new([[w, h]])) & (points >= points.new([[0, 0]]))
    ).all(dim=-1)
    n_images_p, n_points, n_coords = points.shape

    assert n_images_p == n_images
    assert n_coords == 2

    image_i, point_i = is_in_image_bounds.nonzero(as_tuple=True)

    flat_index = image_i * n_points + point_i
    x, y = points.view(-1, 2)[flat_index].T
    color = color.view(-1, 3)[flat_index]

    image[image_i, :, y, x] = color

    return image
