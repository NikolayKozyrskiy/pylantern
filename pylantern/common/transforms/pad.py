from __future__ import annotations

import cv2
from albumentations import DualTransform, PadIfNeeded


class PadToRatio(PadIfNeeded):
    def __init__(self, ratio=1.4):
        super().__init__(border_mode=cv2.BORDER_CONSTANT)
        self.ratio = ratio

    def _get_target_hw(self, h: int, w: int) -> (int, int):
        ratio = h / w

        if ratio > self.ratio:
            w = int(round(h / self.ratio))
        else:
            h = int(round(w * self.ratio))

        return h, w

    def _update_position_params(
        self, h_top: int, h_bottom: int, w_left: int, w_right: int
    ) -> (int, int, int, int):
        if self.position == PadIfNeeded.PositionType.TOP_LEFT:
            h_bottom += h_top
            w_right += w_left
            h_top = 0
            w_left = 0

        elif self.position == PadIfNeeded.PositionType.TOP_RIGHT:
            h_bottom += h_top
            w_left += w_right
            h_top = 0
            w_right = 0

        elif self.position == PadIfNeeded.PositionType.BOTTOM_LEFT:
            h_top += h_bottom
            w_right += w_left
            h_bottom = 0
            w_left = 0

        elif self.position == PadIfNeeded.PositionType.BOTTOM_RIGHT:
            h_top += h_bottom
            w_left += w_right
            h_bottom = 0
            w_right = 0

        return h_top, h_bottom, w_left, w_right

    def update_params(self, params, **kwargs):
        params = DualTransform.update_params(self, params, **kwargs)
        rows = params["rows"]
        cols = params["cols"]
        target_h, target_w = self._get_target_hw(rows, cols)

        if rows < target_h:
            h_pad_top = int((target_h - rows) / 2.0)
            h_pad_bottom = target_h - rows - h_pad_top
        else:
            h_pad_top = 0
            h_pad_bottom = 0

        if cols < target_w:
            w_pad_left = int((target_w - cols) / 2.0)
            w_pad_right = target_w - cols - w_pad_left
        else:
            w_pad_left = 0
            w_pad_right = 0

        (
            h_pad_top,
            h_pad_bottom,
            w_pad_left,
            w_pad_right,
        ) = self._update_position_params(
            h_top=h_pad_top,
            h_bottom=h_pad_bottom,
            w_left=w_pad_left,
            w_right=w_pad_right,
        )

        params.update(
            {
                "pad_top": h_pad_top,
                "pad_bottom": h_pad_bottom,
                "pad_left": w_pad_left,
                "pad_right": w_pad_right,
            }
        )
        return params
