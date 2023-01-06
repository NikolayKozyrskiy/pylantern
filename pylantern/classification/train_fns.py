from collections import defaultdict
from pathlib import Path
from typing import List, Optional
import warnings

import numpy as np
import torch
import torch.nn as nn
from ignite.utils import convert_tensor
from ignite.distributed.auto import auto_dataloader
from ignite.metrics.accumulation import Average
import pandas as pd

from matches.loop import Loop
from matches.shortcuts.optimizer import SchedulerScopeType
from matches.utils import seed_everything, setup_cudnn_reproducibility

from ..common.utils import (
    enumerate_normalized,
    log_optimizer_lrs,
    consume_metric,
    get_device,
)
from ..common.train_utils import predict_dataloader
from .config import ClassificationConfig
from .data.dataloader import get_train_loader, get_validation_loader
from .pipeline import ClassificationPipeline, pipeline_from_config
from .output_dispatcher import OutputDispatcherClr


warnings.filterwarnings("ignore", module="torch.optim.lr_scheduler")


def train_fn(loop: Loop, config: ClassificationConfig) -> None:
    seed_everything(42)
    setup_cudnn_reproducibility(False, True)

    device = get_device()

    train_loader = loop._loader_override(get_train_loader(config), "train")
    valid_loader = loop._loader_override(get_validation_loader(config), "valid")

    pipeline: ClassificationPipeline = pipeline_from_config(config, device)

    optimizer = config.optimizer(pipeline.model)

    out_dispatcher = OutputDispatcherClr(
        loss_aggregation_weigths=config.loss_aggregation_weigths, metrics=config.metrics
    )

    loop.attach(model=pipeline.model, optimizer=optimizer)
    config.resume(loop, pipeline)
    config.postprocess(loop, pipeline)
    scheduler = config.scheduler(optimizer)

    losses_d = defaultdict(lambda: Average(device=device))
    metrics_d = defaultdict(lambda: Average(device=device))

    # if scheduler is not None:
    #     loop.attach(scheduler=scheduler)

    def _train(loop: Loop):
        def handle_batch(batch):
            with pipeline.batch_scope(batch):
                losses = out_dispatcher.compute_losses(pipeline, losses_d)
                metrics = out_dispatcher.compute_metrics(pipeline, metrics_d)
            return losses.aggregated

        # torch.autograd.set_detect_anomaly(True)
        train_eval_batch = None
        for epoch in loop.iterate_epochs(config.max_epoch):
            # Train part
            for epoch_fraction, batch in enumerate_normalized(
                loop.iterate_dataloader(train_loader, "train"), len(train_loader)
            ):
                cur_iter = int(np.round((epoch + epoch_fraction) * len(train_loader)))
                if train_eval_batch is None:
                    train_eval_batch = convert_tensor(
                        batch, device="cpu", non_blocking=True
                    )
                loss = handle_batch(batch)
                loop.backward(loss)
                loop.optimizer_step(optimizer, zero_grad="set_to_none")
                if scheduler is not None:
                    scheduler.step(SchedulerScopeType.BATCH, cur_iter)
                log_optimizer_lrs(loop, optimizer)

                if epoch_fraction == 0.0:
                    # noinspection PyTypeChecker
                    with loop.mode("valid"), pipeline.batch_scope(
                        convert_tensor(
                            train_eval_batch, device=device, non_blocking=True
                        )
                    ):
                        for fn in config.log_vis_fns:
                            fn(loop, pipeline, "train")

            if scheduler is not None:
                scheduler.step(SchedulerScopeType.EPOCH, epoch)

            consume_metric(loop, losses_d, prefix="train")
            consume_metric(loop, metrics_d, prefix="train")

            # Valid part
            for i, batch in enumerate(loop.iterate_dataloader(valid_loader)):
                handle_batch(batch)
                if i == 0:
                    with pipeline.batch_scope(batch):
                        for fn in config.log_vis_fns:
                            fn(loop, pipeline, "valid")
            consume_metric(loop, losses_d, prefix="valid")
            consume_metric(loop, metrics_d, prefix="valid")

        predict_dataloader(
            loop,
            pipeline,
            valid_loader,
            out_dispatcher,
            loop.logdir / "valid_infer",
            verbose=True,
        )

    loop.run(_train)


def infer_fn(
    loop: Loop,
    config: ClassificationConfig,
    checkpoint: str = "best",
    data_root: Optional[Path] = None,
    output_name: Optional[str] = None,
) -> None:
    device = get_device()

    data_root = config.data_root if data_root is None else data_root
    output_name = checkpoint if output_name is None else output_name

    loader = get_validation_loader(config)
    pipeline: ClassificationPipeline = pipeline_from_config(config, device)

    out_dispatcher = OutputDispatcherClr(
        loss_aggregation_weigths=config.loss_aggregation_weigths, metrics=config.metrics
    )

    loop.attach(model=pipeline.model)
    loop.state_manager.read_state(
        loop.logdir / f"{checkpoint}.pth", skip_keys=["optimizer", "scheduler"]
    )

    def _infer(loop: Loop):
        predict_dataloader(
            loop,
            pipeline,
            loader,
            out_dispatcher,
            loop.logdir / output_name,
            verbose=True,
        )

    loop.run(_infer)
    return None
