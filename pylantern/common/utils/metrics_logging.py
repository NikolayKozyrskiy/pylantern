from typing import Dict

import pandas as pd
from ignite.distributed import one_rank_only
from ignite.metrics.accumulation import Average
from matches.loop import IterationType, Loop
from matches.shortcuts.callbacks import get_metrics_summary
from torch.optim import Optimizer


@one_rank_only()
def log_optimizer_lrs(
    loop: Loop,
    optimizer: Optimizer,
    prefix: str = "lr",
    iteration: IterationType = IterationType.AUTO,
) -> None:
    for i, g in enumerate(optimizer.param_groups):
        loop.metrics.log(f"{prefix}/group_{i}", g["lr"], iteration)


@one_rank_only()
def log_optimizer_single_group_lr(
    loop: Loop,
    optimizer: Optimizer,
    optimizer_plot_name: str = "lr/optimizer",
    iteration: IterationType = IterationType.AUTO,
) -> None:
    loop.metrics.log(optimizer_plot_name, optimizer.param_groups[0]["lr"], iteration)


def consume_metric(loop: Loop, avg_dict: Dict[str, Average], prefix: str) -> None:
    for name, value in avg_dict.items():
        loop.metrics.consume(f"{prefix}/{name}", value)


@one_rank_only()
def print_best_metrics_summary(loop: Loop) -> None:
    summary = get_metrics_summary(loop)
    if summary is not None:
        print(f"Metrics summary:\n{pd.DataFrame(summary)}")
    return None
