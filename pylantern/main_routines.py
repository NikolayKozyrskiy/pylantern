from pathlib import Path
from typing import Optional, Type, Callable

from matches.loop import Loop
from matches.accelerators import DDPAccelerator, VanillaAccelerator

from .config import BaseConfig, load_config, dump_config_txt
from .config_generator import (
    ConfigGenerator,
    ConfigGeneratorManager,
    load_config_generator,
)
from .common.utils import (
    copy_config,
    copy_config_generator,
    prepare_comment,
    prepare_logdir,
    DevMode,
    print_best_metrics_summary,
)


def train_routine(
    config_path: Path,
    config_cls: Type[BaseConfig],
    train_fn: Callable,
    comment: Optional[str] = None,
    logdir: Optional[Path] = None,
    gpus: Optional[str] = None,
    dev_mode: DevMode = DevMode.DISABLED,
):
    config: config_cls = load_config(config_path, config_cls)
    comment, config = prepare_comment(comment, config_path, config)
    logdir = prepare_logdir(logdir, comment)
    copy_config(config_path, logdir)

    loop = Loop(
        logdir,
        config.train_callbacks(dev_mode != dev_mode.DISABLED),
        loader_override=dev_mode.value,
    )
    loop.launch(
        train_fn,
        DDPAccelerator(gpus) if gpus is not None else VanillaAccelerator("cpu"),
        config=config,
    )
    print_best_metrics_summary(loop)


def train_config_generator_routine(
    config_generator_path: Path,
    root_log_dir: Path,
    train_fn: Callable,
    comment_postfix: Optional[str] = None,
    gpus: Optional[str] = None,
    dev_mode: DevMode = DevMode.DISABLED,
):
    root_log_dir.mkdir(exist_ok=True, parents=True)
    copy_config_generator(config_generator_path, root_log_dir)
    config_generator: ConfigGenerator = load_config_generator(config_generator_path)
    copy_config(config_generator.base_config_path, root_log_dir)
    config_generator_manager = ConfigGeneratorManager(config_generator)

    for config_idx, config in enumerate(config_generator()):
        comment = f"{config_idx:02d}__{config.comment}"
        if comment_postfix is not None:
            comment += f"_{comment_postfix}"
        config.comment = comment

        logdir = root_log_dir / comment
        logdir.mkdir(exist_ok=True, parents=True)
        dump_config_txt(config, logdir / "config")
        # dump_config_json(config, logdir / "config.json")
        # dump_pickle(config, logdir / "config.pkl")

        loop = Loop(
            logdir,
            config.train_callbacks(dev_mode != dev_mode.DISABLED),
            loader_override=dev_mode.value,
        )
        loop.launch(
            train_fn,
            DDPAccelerator(gpus) if gpus is not None else VanillaAccelerator("cpu"),
            config=config,
        )

        config_generator_manager.update(loop, config_idx)

    config_generator_manager.finalize(root_log_dir)


def infer_routine(
    config_path: Path,
    config_cls: Type[BaseConfig],
    infer_fn: Callable,
    logdir: Optional[Path] = None,
    checkpoint: str = "best",
    data_root: Optional[Path] = None,
    output_name: Optional[str] = None,
    gpus: Optional[str] = None,
):
    config: config_cls = load_config(config_path, config_cls)
    if logdir is None:
        logdir = config_path.parent
    loop = Loop(
        logdir,
        config.valid_callbacks(),
    )
    loop.launch(
        infer_fn,
        DDPAccelerator(gpus) if gpus is not None else VanillaAccelerator("cpu"),
        config=config,
        checkpoint=checkpoint,
        data_root=data_root,
        output_name=output_name,
    )


def infer_config_generator_routine(
    config_generator_path: Path,
    root_log_dir: Path,
    infer_fn: Callable,
    data_root: Optional[Path] = None,
    output_name: Optional[str] = None,
    gpus: Optional[str] = None,
):
    config_generator: ConfigGenerator = load_config_generator(config_generator_path)
    if root_log_dir is None:
        root_log_dir = config_generator_path.parent

    # TODO: implemet search for best config idx
    best_idx = 0

    config = config_generator.get_config(best_idx)

    loop = Loop(
        root_log_dir,
        config.valid_callbacks(),
    )
    loop.launch(
        infer_fn,
        DDPAccelerator(gpus) if gpus is not None else VanillaAccelerator("cpu"),
        config=config,
        checkpoint=checkpoint,
        data_root=data_root,
        output_name=output_name,
    )
