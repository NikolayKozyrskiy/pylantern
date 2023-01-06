from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, List, Tuple, Type, Callable

from torch.utils.data import DataLoader
import pandas as pd
from matches.loop import Loop
from matches.accelerators import DDPAccelerator, VanillaAccelerator

from ..config import BaseConfig, load_config, dump_config_txt
from ..config_generator import ConfigGenerator, load_config_generator
from ..pipeline import Pipeline
from ..output_dispatcher import OutputDispatcher, filter_and_uncollate
from .utils import (
    copy_config,
    copy_config_generator,
    prepare_comment,
    prepare_logdir,
    DevMode,
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

    for idx, config in enumerate(config_generator()):
        comment = f"{idx:02d}__{config.comment}"
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


def predict_dataloader(
    loop: Loop,
    pipeline: Pipeline,
    dataloader: DataLoader,
    out_dispatcher: OutputDispatcher,
    save_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[List[dict], List[dict]]:
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
        if verbose:
            print(mean)
        (save_dir / f"{name}_mean.txt").write_text(str(mean))
        return None

    _dump(losses, "losses")
    _dump(metrics, "metrics")

    return losses, metrics
