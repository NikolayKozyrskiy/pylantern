from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import ignite.distributed as idist
from ignite.utils import convert_tensor
from ignite.distributed.auto import auto_dataloader
from ignite.metrics.accumulation import Average
import pandas as pd

from matches.loop import Loop
from matches.loop.loader_scheduling import DataloaderSchedulerWrapper
from matches.shortcuts.optimizer import SchedulerScopeType
from matches.utils import seed_everything, setup_cudnn_reproducibility

from ..common.utils import enumerate_normalized, log_optimizer_lrs
from ..output_dispatcher import filter_and_uncollate

from .config import ClassificationConfig
from .data import get_train_loader, get_validation_loader
from .pipeline import ClassificationPipeline, pipeline_from_config
from .output_dispatcher import OutputDispatcherClr


warnings.filterwarnings("ignore", module="torch.optim.lr_scheduler")


def train_fn(loop: Loop, config: ClassificationConfig):
    seed_everything(42)
    setup_cudnn_reproducibility(False, True)

    device = f"cuda:{torch.cuda.current_device()}"

    train_loader = loop._loader_override(get_train_loader(config), "train")
    valid_loader = loop._loader_override(get_validation_loader(config), "valid")

    pipeline = pipeline_from_config(config, device)

    optimizer = config.optimizer(pipeline.ae)

    out_dispatcher = OutputDispatcherClr(
        loss_aggregation_weigths=config.loss_aggregation_weigths, metrics=config.metrics
    )

    loop.attach(ae_model=pipeline.ae, ae_optimizer=optimizer)
    config.resume(loop)
    config.postprocess(loop, pipeline)
    scheduler = config.scheduler(optimizer)

    losses_d = defaultdict(lambda: Average(device=device))
    metrics_d = defaultdict(lambda: Average(device=device))

    # if scheduler is not None:
    #     loop.attach(ae_scheduler=scheduler)

    def stage_1(loop: Loop):
        def handle_batch(batch):
            with pipeline.batch_scope(batch):
                losses = out_dispatcher.compute_losses(pipeline)
                metrics = out_dispatcher.compute_metrics(pipeline)

            for name, v in losses.computed_values.items():
                losses_d[name].update(v.detach().mean())
            for name, v in metrics.computed_values.items():
                metrics_d[name].update(v.detach().mean())

            return losses.aggregated

        # torch.autograd.set_detect_anomaly(True)
        train_eval_batch = None
        for epoch in loop.iterate_epochs(config.max_epoch):
            # Train part
            for epoch_fraction, batch in enumerate_normalized(
                loop.iterate_dataloader(train_loader, "train"), len(train_loader)
            ):
                if train_eval_batch is None:
                    train_eval_batch = convert_tensor(
                        batch, device="cpu", non_blocking=True
                    )

                cur_iter = int((epoch + epoch_fraction) * len(train_loader))
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
                        log_images_to_tb_batch(loop, pipeline, "train")
                        log_embeddings_to_tb_batch(loop, pipeline, "train")

            if scheduler is not None:
                scheduler.step(SchedulerScopeType.EPOCH, epoch)

            for name, value in losses_d.items():
                loop.metrics.consume(f"train/{name}", value)
            for name, value in metrics_d.items():
                loop.metrics.consume(f"train/{name}", value)

            # Valid part
            for i, batch in enumerate(loop.iterate_dataloader(valid_loader)):
                handle_batch(batch)
                if i == 0:
                    with pipeline.batch_scope(batch):
                        for fn in config.log_vis_fns:
                            fn(loop, pipeline, "valid")
            for name, value in losses_d.items():
                loop.metrics.consume(f"valid/{name}", value)
            for name, value in metrics_d.items():
                loop.metrics.consume(f"valid/{name}", value)

        predict_dataloader(loop, pipeline, valid_loader, out_dispatcher)

    loop.run(stage_1)


def infer_fn(
    loop: Loop,
    config: ClassificationConfig,
    checkpoint: str = "best",
    data_root: Optional[Path] = None,
    output_name: Optional[str] = None,
):
    device = f"cuda:{torch.cuda.current_device()}"

    data_root = config.data_root if data_root is None else data_root
    output_name = checkpoint if output_name is None else output_name

    loader = get_validation_loader(config)
    pipeline = pipeline_from_config(config, device)

    out_dispatcher = OutputDispatcherClr(
        loss_aggregation_weigths=config.loss_aggregation_weigths, metrics=config.metrics
    )

    loop.attach(ae_model=pipeline.ae)
    loop.state_manager.read_state(loop.logdir / f"{checkpoint}.pth")

    def infer(loop: Loop):
        predict_dataloader(
            loop, pipeline, loader, out_dispatcher, loop.logdir / output_name
        )

    loop.run(infer)
    return None


def predict_dataloader(
    loop: Loop,
    pipeline: ClassificationPipeline,
    dataloader: DataLoader,
    out_dispatcher: OutputDispatcherClr,
    save_dir: Optional[Path] = None,
):
    if save_dir is None:
        save_dir = loop.logdir / "default_infer"

    save_dir.mkdir(parents=True, exist_ok=True)

    losses, metrics = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for batch in loop.iterate_dataloader(dataloader):
            with pipeline.batch_scope(batch):

                for f in pipeline.config.output_config:
                    f(
                        pipeline,
                        pool,
                        save_dir,
                        name_postfix=str(save_dir).split("/")[1],
                    )

                losses.extend(
                    filter_and_uncollate(
                        out_dispatcher.compute_losses(pipeline).computed_values,
                        pipeline,
                    )
                )
                metrics.extend(
                    filter_and_uncollate(
                        out_dispatcher.compute_metrics(pipeline).computed_values,
                        pipeline,
                    )
                )

    def _dump(data: List, name: str):
        data = pd.DataFrame(data)
        data.to_csv(save_dir / f"{name}.csv")
        mean = data.mean(numeric_only=True)
        print(mean)
        (save_dir / f"{name}_mean.txt").write_text(str(mean))
        return None

    _dump(losses, "losses")
    _dump(metrics, "metrics")
