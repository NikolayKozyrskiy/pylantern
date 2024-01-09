import warnings
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Type

from ignite.distributed import auto_model
from ignite.metrics.accumulation import Average
from ignite.utils import convert_tensor
from matches.loop import Loop
from matches.utils import seed_everything, setup_cudnn_reproducibility

from pylantern.common.train_fns import predict_dataloader
from pylantern.common.utils import get_device
from pylantern.common.utils.metrics_logging import (
    consume_metric,
    log_optimizer_single_group_lr,
)
from pylantern.config import load_config
from pylantern.common.visualization.image.wandb import log_images_to_wandb
from pylantern.tasks.gan.pix2pix.data.dataloaders import (
    get_train_loader,
    get_validation_loader,
)
from pylantern.tasks.gan.pix2pix.output_dispatchers import FacePix2PixHDOutputDispatcher
from pylantern.tasks.gan.pix2pix.pipelines import (
    FacePix2PixHDPipeline,
    face_pix2pixhd_pipeline_from_config,
)

if TYPE_CHECKING:
    from pylantern.tasks.gan.pix2pix.configs import FacePix2PixHDConfig

warnings.filterwarnings("ignore", module="torch.optim.lr_scheduler")
warnings.simplefilter("ignore")


def train_face_pix2pixhd_fn(
    loop: Loop,
    config_path: Path,
    config_cls: Type["FacePix2PixHDConfig"],
) -> None:
    config: "FacePix2PixHDConfig" = load_config(config_path, config_cls)
    seed_everything(42)
    setup_cudnn_reproducibility(deterministic=False, benchmark=True)
    device = get_device()
    train_loader = loop._loader_override(get_train_loader(config), "train")
    valid_loader = loop._loader_override(get_validation_loader(config), "valid")

    pipeline: "FacePix2PixHDPipeline" = face_pix2pixhd_pipeline_from_config(
        config, device
    )
    config.preprocess(loop, pipeline)

    pipeline.generator_model = auto_model(pipeline.generator_model)
    pipeline.discriminator_model = auto_model(pipeline.discriminator_model)
    optimizer_generator = config.optimizer_generator(pipeline.generator_model)
    optimizer_discriminator = config.optimizer_discriminator(
        pipeline.discriminator_model
    )
    scheduler_generator = config.scheduler_generator(optimizer_generator)
    scheduler_discriminator = config.scheduler_discriminator(optimizer_discriminator)
    loop.attach(
        generator_model=pipeline.generator_model,
        discriminator_model=pipeline.discriminator_model,
        optimizer_generator=optimizer_generator,
        optimizer_discriminator=optimizer_discriminator,
        scheduler_generator=scheduler_generator,
        scheduler_discriminator=scheduler_discriminator,
    )

    if pipeline.discriminator_left_eye is not None:
        pipeline.discriminator_left_eye = auto_model(pipeline.discriminator_left_eye)
        optimizer_discriminator_left_eye = config.optimizer_discriminator_left_eye(
            pipeline.discriminator_left_eye
        )
        scheduler_discriminator_left_eye = config.scheduler_discriminator_left_eye(
            pipeline.discriminator_left_eye
        )
        loop.attach(
            discriminator_left_eye=pipeline.discriminator_left_eye,
            optimizer_discriminator_left_eye=optimizer_discriminator_left_eye,
            scheduler_discriminator_left_eye=scheduler_discriminator_left_eye,
        )
    else:
        optimizer_discriminator_left_eye = None
        scheduler_discriminator_left_eye = None
    if pipeline.discriminator_right_eye is not None:
        pipeline.discriminator_right_eye = auto_model(pipeline.discriminator_right_eye)
        optimizer_discriminator_right_eye = config.optimizer_discriminator_right_eye(
            pipeline.discriminator_right_eye
        )
        scheduler_discriminator_right_eye = config.scheduler_discriminator_right_eye(
            pipeline.discriminator_right_eye
        )
        loop.attach(
            discriminator_right_eye=pipeline.discriminator_right_eye,
            optimizer_discriminator_right_eye=optimizer_discriminator_right_eye,
            scheduler_discriminator_right_eye=scheduler_discriminator_right_eye,
        )
    else:
        optimizer_discriminator_right_eye = None
        scheduler_discriminator_right_eye = None
    if pipeline.discriminator_mouth is not None:
        pipeline.discriminator_mouth = auto_model(pipeline.discriminator_mouth)
        optimizer_discriminator_mouth = config.optimizer_discriminator_mouth(
            pipeline.discriminator_mouth
        )
        scheduler_discriminator_mouth = config.scheduler_discriminator_mouth(
            pipeline.discriminator_mouth
        )
        loop.attach(
            discriminator_mouth=pipeline.discriminator_mouth,
            optimizer_discriminator_mouth=optimizer_discriminator_mouth,
            scheduler_discriminator_mouth=scheduler_discriminator_mouth,
        )
    else:
        optimizer_discriminator_mouth = None
        scheduler_discriminator_mouth = None

    out_dispatcher = FacePix2PixHDOutputDispatcher(config=config, device=device)

    config.resume(loop, pipeline)

    losses_train_d = defaultdict(lambda: Average(device=device))
    metrics_train_d = defaultdict(lambda: Average(device=device))
    metrics_valid_d = defaultdict(lambda: Average(device=device))

    config.postprocess(loop, pipeline)

    def _train(loop: Loop):
        # torch.autograd.set_detect_anomaly(True)
        train_eval_batch = None
        for epoch in loop.iterate_epochs(config.max_epoch):
            # Train part
            for iter_idx, batch in enumerate(
                loop.iterate_dataloader(train_loader, "train")
            ):
                if train_eval_batch is None:
                    train_eval_batch = convert_tensor(
                        batch, device="cpu", non_blocking=True
                    )
                with pipeline.batch_scope(batch):
                    loss_G = out_dispatcher.compute_losses_group(
                        "generator",
                        pipeline=pipeline,
                        loop=loop,
                        losses_avg_dict=losses_train_d,
                    ).aggregated
                    loss_D = out_dispatcher.compute_losses_group(
                        "discriminator",
                        pipeline=pipeline,
                        loop=loop,
                        losses_avg_dict=losses_train_d,
                    ).aggregated
                    if out_dispatcher.has_group("component_left_eye"):
                        loss_D_left_eye = out_dispatcher.compute_losses_group(
                            "component_left_eye",
                            pipeline=pipeline,
                            loop=loop,
                            losses_avg_dict=losses_train_d,
                        ).aggregated
                    if out_dispatcher.has_group("component_right_eye"):
                        loss_D_right_eye = out_dispatcher.compute_losses_group(
                            "component_right_eye",
                            pipeline=pipeline,
                            loop=loop,
                            losses_avg_dict=losses_train_d,
                        ).aggregated
                    if out_dispatcher.has_group("component_mouth"):
                        loss_D_mouth = out_dispatcher.compute_losses_group(
                            "component_mouth",
                            pipeline=pipeline,
                            loop=loop,
                            losses_avg_dict=losses_train_d,
                        ).aggregated
                    out_dispatcher.compute_metrics(
                        pipeline=pipeline,
                        loop=loop,
                        metrics_avg_dict=metrics_train_d,
                    )

                loop.zero_grad_backward_step(
                    loss=loss_G,
                    optimizer=optimizer_generator,
                    set_to_none=True,
                )
                scheduler_generator.step_batch(iter_idx)
                log_optimizer_single_group_lr(
                    loop, optimizer_generator, optimizer_plot_name="lr/generator"
                )

                loop.zero_grad_backward_step(
                    loss=loss_D,
                    optimizer=optimizer_discriminator,
                    set_to_none=True,
                )
                scheduler_discriminator.step_batch(iter_idx)
                log_optimizer_single_group_lr(
                    loop,
                    optimizer_discriminator,
                    optimizer_plot_name="lr/discriminator",
                )

                if out_dispatcher.has_group("component_left_eye"):
                    loop.zero_grad_backward_step(
                        loss=loss_D_left_eye,
                        optimizer=optimizer_discriminator_left_eye,
                        set_to_none=True,
                    )
                    scheduler_discriminator_left_eye.step_batch(iter_idx)
                    log_optimizer_single_group_lr(
                        loop,
                        optimizer_discriminator_left_eye,
                        optimizer_plot_name="lr/discriminator_left_eye",
                    )

                if out_dispatcher.has_group("component_right_eye"):
                    loop.zero_grad_backward_step(
                        loss=loss_D_right_eye,
                        optimizer=optimizer_discriminator_right_eye,
                        set_to_none=True,
                    )
                    scheduler_discriminator_right_eye.step_batch(iter_idx)
                    log_optimizer_single_group_lr(
                        loop,
                        optimizer_discriminator_right_eye,
                        optimizer_plot_name="lr/discriminator_right_eye",
                    )

                if out_dispatcher.has_group("component_mouth"):
                    loop.zero_grad_backward_step(
                        loss=loss_D_mouth,
                        optimizer=optimizer_discriminator_mouth,
                        set_to_none=True,
                    )
                    scheduler_discriminator_mouth.step_batch(iter_idx)
                    log_optimizer_single_group_lr(
                        loop,
                        optimizer_discriminator_mouth,
                        optimizer_plot_name="lr/discriminator_mouth",
                    )

                if iter_idx == 0:
                    # noinspection PyTypeChecker
                    with loop.mode("valid"), pipeline.batch_scope(
                        convert_tensor(
                            train_eval_batch, device=device, non_blocking=True
                        )
                    ):
                        log_images_to_wandb(loop, pipeline, prefix="train")
            scheduler_generator.step_epoch(epoch)
            scheduler_discriminator.step_epoch(epoch)
            if pipeline.discriminator_left_eye is not None:
                scheduler_discriminator_left_eye.step_epoch(epoch)
            if pipeline.discriminator_right_eye is not None:
                scheduler_discriminator_right_eye.step_epoch(epoch)
            if pipeline.discriminator_mouth is not None:
                scheduler_discriminator_mouth.step_epoch(epoch)
            consume_metric(loop, losses_train_d, prefix="train")
            consume_metric(loop, metrics_train_d, prefix="train")

            # Valid part
            for i, batch in enumerate(loop.iterate_dataloader(valid_loader)):
                with pipeline.batch_scope(batch):
                    out_dispatcher.compute_metrics(
                        pipeline=pipeline,
                        loop=loop,
                        metrics_avg_dict=metrics_valid_d,
                    )
                    if i == 0:
                        log_images_to_wandb(loop, pipeline, prefix="valid")
            consume_metric(loop, metrics_valid_d, prefix="valid")

        predict_dataloader(
            loop=loop,
            pipeline=pipeline,
            dataloader=valid_loader,
            output_dispatcher=out_dispatcher,
            group_losses=None,
            save_dir=loop.logdir / "valid_infer",
            verbose=True,
        )

    loop.run(_train)


def test_face_pix2pixhd_fn(
    loop: Loop,
    config: "FacePix2PixHDConfig",
    checkpoint: str = "best",
    data_root: Optional[Path] = None,
    output_name: Optional[str] = None,
) -> None:
    device = get_device()

    data_root = config.root_path if data_root is None else data_root
    output_name = checkpoint if output_name is None else output_name

    loader = get_validation_loader(config)
    pipeline: "FacePix2PixHDPipeline" = face_pix2pixhd_pipeline_from_config(
        config, device
    )

    out_dispatcher = FacePix2PixHDOutputDispatcher(config=config, device=device)

    loop.attach(generator_model=pipeline.generator_model)
    loop.state_manager.read_state(
        loop.logdir / f"{checkpoint}.pth", skip_keys=["optimizer", "scheduler"]
    )

    def _infer(loop: Loop):
        predict_dataloader(
            loop=loop,
            pipeline=pipeline,
            dataloader=loader,
            out_dispatcher=out_dispatcher,
            group_losses=None,
            save_dir=loop.logdir / output_name,
            verbose=True,
        )

    loop.run(_infer)
    return None
