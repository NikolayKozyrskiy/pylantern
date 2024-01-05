from pathlib import Path
from typing import Callable, Optional, Type

from matches.accelerators import DDPAccelerator, VanillaAccelerator
from matches.loop import Loop

from .common.utils import (
    DevMode,
    copy_config,
    copy_config_generator,
    prepare_comment,
    prepare_logdir,
    wrap_tqdm,
)
from .config import BaseConfig, dump_config_dict, load_config
from .config_generator import (
    GENERATOR_SUMMARY_FILE,
    ConfigGenerator,
    ConfigGeneratorManager,
    load_config_generator,
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
        DDPAccelerator(gpus),
        config_path=config_path,
        config_cls=config_cls,
    )


def train_config_generator_routine(
    config_generator_path: Path,
    root_log_dir: Path,
    train_fn: Callable,
    gpus: Optional[str] = None,
    dev_mode: DevMode = DevMode.DISABLED,
):
    root_log_dir.mkdir(exist_ok=True, parents=True)
    copy_config_generator(config_generator_path, root_log_dir)
    config_generator: "ConfigGenerator" = load_config_generator(config_generator_path)
    copy_config(config_generator.base_config_path, root_log_dir)
    config_generator_manager = ConfigGeneratorManager(config_generator)

    for config_idx, config in enumerate(
        wrap_tqdm(config_generator(), "Generator", len(config_generator))
    ):
        logdir = root_log_dir / config.comment
        logdir.mkdir(exist_ok=True, parents=True)
        dump_config_dict(config, logdir / "config.json")

        loop = Loop(
            logdir,
            config.train_callbacks(dev_mode != dev_mode.DISABLED),
            loader_override=dev_mode.value,
        )
        loop.launch(
            train_fn,
            DDPAccelerator(gpus),
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
        DDPAccelerator(gpus),
        config=config,
        checkpoint=checkpoint,
        data_root=data_root,
        output_name=output_name,
    )


def infer_config_generator_routine(
    config_generator_path: Path,
    infer_fn: Callable,
    root_log_dir: Optional[Path] = None,
    summary_path: Optional[Path] = None,
    metric_name: Optional[str] = None,
    checkpoint: str = "best",
    data_root: Optional[Path] = None,
    output_name: Optional[str] = None,
    gpus: Optional[str] = None,
):
    config_generator: ConfigGenerator = load_config_generator(config_generator_path)
    if root_log_dir is None:
        root_log_dir = config_generator_path.parent
    config_generator_manager = ConfigGeneratorManager(config_generator)
    if summary_path is None:
        summary_path = root_log_dir / GENERATOR_SUMMARY_FILE
    config = config_generator_manager.get_best_config(
        summary_path, metric_name=metric_name
    )
    logdir = root_log_dir / config.comment

    loop = Loop(
        logdir,
        config.valid_callbacks(),
    )
    loop.launch(
        infer_fn,
        DDPAccelerator(gpus),
        config=config,
        checkpoint=checkpoint,
        data_root=data_root,
        output_name=output_name,
    )
