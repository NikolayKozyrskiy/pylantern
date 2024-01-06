import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional, Type

from ignite.distributed import auto_model
from ignite.metrics.accumulation import Average
from ignite.utils import convert_tensor
from matches.loop import Loop
from matches.utils import seed_everything, setup_cudnn_reproducibility

from pylantern.common.train_utils import predict_dataloader
from pylantern.common.utils import get_device
from pylantern.common.utils.metrics_logging import consume_metric, log_optimizer_lrs
from pylantern.config import load_config
from pylantern.tasks.gan.common.visualization.wandb import log_images_to_wandb

from ..configs import BasePix2PixConfig
from ..data.dataloaders import get_train_loader, get_validation_loader
from ..output_dispatcher import GanPix2PixOutputDispatcher
from ..pipelines import BasePix2PixPipeline, pipeline_from_config

warnings.filterwarnings("ignore", module="torch.optim.lr_scheduler")
warnings.simplefilter("ignore")


def train_baseline_fn(
    loop: Loop,
    config_path: Path,
    config_cls: Type[BasePix2PixConfig],
) -> None:
    config: "BasePix2PixConfig" = load_config(config_path, config_cls)
    seed_everything(42)
    setup_cudnn_reproducibility(deterministic=False, benchmark=True)

    device = get_device()

    train_loader = loop._loader_override(get_train_loader(config), "train")
    valid_loader = loop._loader_override(get_validation_loader(config), "valid")

    pipeline: "BasePix2PixPipeline" = pipeline_from_config(config, device)
    config.preprocess(loop, pipeline)
    pipeline.face_swapper = auto_model(pipeline.face_swapper)

    out_dispatcher = GanPix2PixOutputDispatcher(config, device=device)
    optimizer = config.optimizer(pipeline.face_swapper)
    scheduler = config.scheduler(optimizer)

    loop.attach(
        face_swapper=pipeline.face_swapper, optimizer=optimizer, scheduler=scheduler
    )

    config.resume(loop, pipeline)
    config.postprocess(loop, pipeline)

    losses_train_d = defaultdict(lambda: Average(device=device))
    losses_valid_d = defaultdict(lambda: Average(device=device))
    metrics_train_d = defaultdict(lambda: Average(device=device))
    metrics_valid_d = defaultdict(lambda: Average(device=device))

    def _train(loop: Loop):
        def handle_batch_train(batch):
            with pipeline.batch_scope(batch):
                losses = out_dispatcher.compute_losses(
                    pipeline, loop=loop, losses_avg_dict=losses_train_d
                )
                out_dispatcher.compute_metrics(
                    pipeline, loop=loop, metrics_avg_dict=metrics_train_d
                )
            return losses.aggregated

        def handle_batch_valid(batch):
            with pipeline.batch_scope(batch):
                out_dispatcher.compute_losses(
                    pipeline, loop=loop, losses_avg_dict=losses_valid_d
                )
                out_dispatcher.compute_metrics(
                    pipeline, loop=loop, metrics_avg_dict=metrics_valid_d
                )
            return None

        # torch.autograd.set_detect_anomaly(True)
        train_eval_batch = None
        for epoch in loop.iterate_epochs(config.max_epoch):
            # Train part
            for iter_idx, batch in enumerate(
                loop.iterate_dataloader(train_loader, "train"), len(train_loader)
            ):
                if train_eval_batch is None:
                    train_eval_batch = convert_tensor(
                        batch, device="cpu", non_blocking=True
                    )
                loss = handle_batch_train(batch)
                loop.backward(loss)
                loop.optimizer_step(optimizer, zero_grad="set_to_none")
                scheduler.step_batch(iter_idx)
                log_optimizer_lrs(loop, optimizer)

                if iter_idx == 0:
                    # noinspection PyTypeChecker
                    with loop.mode("valid"), pipeline.batch_scope(
                        convert_tensor(
                            train_eval_batch, device=device, non_blocking=True
                        )
                    ):
                        log_images_to_wandb(loop, pipeline, prefix="train")

            scheduler.step_epoch(epoch)

            consume_metric(loop, losses_train_d, prefix="train")
            consume_metric(loop, metrics_train_d, prefix="train")

            # Valid part
            for i, batch in enumerate(loop.iterate_dataloader(valid_loader)):
                handle_batch_valid(batch)
                if i == 0:
                    with pipeline.batch_scope(batch):
                        log_images_to_wandb(loop, pipeline, prefix="valid")
            consume_metric(loop, losses_valid_d, prefix="valid")
            consume_metric(loop, metrics_valid_d, prefix="valid")

        predict_dataloader(
            loop,
            pipeline,
            valid_loader,
            out_dispatcher,
            loop.logdir / "valid_infer",
            verbose=True,
        )

    loop.run(_train)


def infer_baseline_fn(
    loop: Loop,
    config: "BasePix2PixConfig",
    checkpoint: str = "best",
    data_root: Optional[Path] = None,
    output_name: Optional[str] = None,
) -> None:
    device = get_device()

    data_root = config.data_root if data_root is None else data_root
    output_name = checkpoint if output_name is None else output_name

    loader = get_validation_loader(config)
    pipeline: "BasePix2PixPipeline" = pipeline_from_config(config, device)

    out_dispatcher = GanPix2PixOutputDispatcher(config=config, device=device)

    loop.attach(face_swapper=pipeline.face_swapper)
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
