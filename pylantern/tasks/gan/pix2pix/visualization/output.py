from __future__ import annotations

from collections import defaultdict
from concurrent.futures import Executor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from pylantern.common.utils import append_json
from pylantern.common.visualization.utils import delayed

if TYPE_CHECKING:
    from pylantern.tasks.gan.pix2pix.pipelines import BasePix2PixPipeline


@delayed
def save_output_statistics(
    pipeline: "BasePix2PixPipeline",
    io_pool: Executor,
    root: Path,
    name_postfix: str = "",
):
    output_file_path = root / "statistics.json"
    statistics = defaultdict(list)

    def _merge_fn(d1: dict, d2: dict):
        for k in d1.keys():
            d1[k] += d2[k]

    with torch.no_grad():
        imgs = pipeline.get_predicted_image()
        statistics["mean"].append(
            imgs.mean(dim=(1, 2, 3)).mean().cpu().detach().tolist()
        )
        statistics["std"].append(imgs.std(dim=(1, 2, 3)).mean().cpu().detach().tolist())
        statistics["max_val"].append(
            torch.amax(imgs, dim=(1, 2, 3)).mean().cpu().detach().tolist()
        )
        statistics["min_val"].append(
            torch.amin(imgs, dim=(1, 2, 3)).mean().cpu().detach().tolist()
        )

        statistics["median"].append(
            np.median(imgs.cpu().detach().numpy(), axis=(1, 2, 3)).mean().tolist()
        )
        statistics["q_999"].append(
            np.quantile(imgs.cpu().detach().numpy(), q=0.999, axis=(1, 2, 3))
            .mean()
            .tolist()
        )
        statistics["q_99"].append(
            np.quantile(imgs.cpu().detach().numpy(), q=0.99, axis=(1, 2, 3))
            .mean()
            .tolist()
        )
        statistics["q_98"].append(
            np.quantile(imgs.cpu().detach().numpy(), q=0.98, axis=(1, 2, 3))
            .mean()
            .tolist()
        )

        statistics["q_001"].append(
            np.quantile(imgs.cpu().detach().numpy(), q=0.001, axis=(1, 2, 3))
            .mean()
            .tolist()
        )
        statistics["q_01"].append(
            np.quantile(imgs.cpu().detach().numpy(), q=0.01, axis=(1, 2, 3))
            .mean()
            .tolist()
        )
        statistics["q_02"].append(
            np.quantile(imgs.cpu().detach().numpy(), q=0.02, axis=(1, 2, 3))
            .mean()
            .tolist()
        )

        append_json(statistics, output_file_path, _merge_fn, indent=2)
