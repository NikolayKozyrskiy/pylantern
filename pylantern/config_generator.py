from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Union, List, Type, Optional
from math import log10, ceil

from matches.loop import Loop
from matches.callbacks import BestModelSaver, BestMetricsReporter
from matches.shortcuts.callbacks import (
    get_callback,
    get_best_model_metric_setup,
    get_metric_best_setups,
)
from matches.shortcuts.metrics import MetricBestSetup
from matches.utils import dump_json
import pandas as pd

from .config import BaseConfig, load_config

PATH_STR = Union[Path, str]
NUM_STR = Union[str, int, float]
DICT_STR_NUMSTR = Dict[str, NUM_STR]

GENERATOR_SUMMARY_FILE = "generator_summary"


class ConfigGenerator:
    def __init__(
        self,
        base_config_path: PATH_STR,
        base_config_class: Type[BaseConfig],
        variable_parameters: Dict[str, List[Any]],
        comment_postfix: Optional[str] = None,
    ) -> None:
        self.base_config_path = base_config_path
        self.base_config_class = base_config_class
        self.variable_parameters = variable_parameters

        self.comment_postfix = "" if comment_postfix is None else f"_{comment_postfix}"
        self.base_config = load_config(base_config_path, base_config_class)
        self._variable_attrs_product = list(dict_cross_product(variable_parameters))
        self._idx_width = ceil(log10(self.__len__() + 1))

    def __call__(self, *args: Any, **kwargs: Any) -> Generator[BaseConfig, None, None]:
        return self.iterate_configs(*args, **kwargs)

    def __len__(self) -> int:
        return len(self._variable_attrs_product)

    def preprocess(self, *args: Any, **kwargs: Any):
        pass

    def postprocess(self, *args: Any, **kwargs: Any):
        pass

    def iterate_configs(
        self, *args: Any, **kwargs: Any
    ) -> Generator[BaseConfig, None, None]:
        for config_idx, new_attrs_dict in enumerate(self._variable_attrs_product):
            config = self._get_compiled_config(new_attrs_dict, config_idx)
            yield config

    def get_config(self, config_idx: int) -> BaseConfig:
        return self._get_compiled_config(
            self._variable_attrs_product[config_idx], config_idx
        )

    def _get_compiled_config(
        self, new_attrs_dict: Dict[str, Any], config_idx: int
    ) -> BaseConfig:
        config = self.base_config.copy(deep=True)
        for k, v in new_attrs_dict.items():
            config.__setattr__(k, v)
        comment = (
            f"{config_idx:0{self._idx_width}d}__{config.comment}{self.comment_postfix}"
        )
        config.__setattr__("comment", comment)
        return config


class MetricBestEntity:
    def __init__(
        self,
        name: str,
        mode: str,
    ) -> None:
        self.name = name
        self.mode = mode

        self.best_config_idx = None
        self.best_epoch_idx = None
        self.total_epochs_num = None
        if mode == "min":
            self.compare_op = min
            self.best_value = float("+inf")
        else:
            self.compare_op = max
            self.best_value = float("-inf")

    def update(self, metric_setup: MetricBestSetup, config_idx: int) -> None:
        better_value = self.compare_op(self.best_value, metric_setup.best_value)
        if better_value != self.best_value:
            self.best_config_idx = config_idx
            self.best_value = better_value
            self.best_epoch_idx = metric_setup.best_epoch_idx
            self.total_epochs_num = metric_setup.total_epochs_num

    def to_dict(self) -> DICT_STR_NUMSTR:
        return {
            "metric_name": self.name,
            "best_config_idx": self.best_config_idx,
            "best_value": self.best_value,
            "best_epoch_idx": self.best_epoch_idx,
            "total_epochs_num": self.total_epochs_num,
        }


DICT_STR_MBE = Dict[str, MetricBestEntity]


class GeneratorSummary:
    summary: Optional[DICT_STR_NUMSTR] = None
    summary_df: Optional[pd.DataFrame] = None

    def make_summary(self, metric_entities_dict: DICT_STR_MBE) -> None:
        self.summary = defaultdict(list)
        for name, metric_best_entity in metric_entities_dict.items():
            self.summary["metric_name"].append(name)
            self.summary["best_config_idx"].append(metric_best_entity.best_config_idx)
            self.summary["best_value"].append(metric_best_entity.best_value)
            self.summary["best_epoch_idx"].append(metric_best_entity.best_epoch_idx)
            self.summary["total_epochs_num"].append(metric_best_entity.total_epochs_num)
        self.summary_df = pd.DataFrame(self.summary)

    def save_summary(self, logdir: PATH_STR) -> None:
        logdir = Path(logdir)
        self.summary_df.to_csv(logdir / f"{GENERATOR_SUMMARY_FILE}.csv")
        (logdir / f"{GENERATOR_SUMMARY_FILE}.txt").write_text(str(self.summary_df))
        dump_json(self.summary, logdir / f"{GENERATOR_SUMMARY_FILE}.json", indent=2)

    def load_summary(self, file_path: PATH_STR) -> None:
        file_path = file_path if str(file_path).endswith(".csv") else f"{file_path}.csv"
        self.summary_df = pd.read_csv(file_path).drop(
            "Unnamed: 0", axis=1, errors="ignore"
        )

    def best_config_idx(self, metric_name: str) -> Optional[int]:
        if self.summary_df is None:
            return None
        config_idx = None
        try:
            config_idx = self.summary_df[
                self.summary_df["metric_name"] == metric_name
            ].best_config_idx.values[0]
            return config_idx
        except KeyError:
            raise ValueError(
                f"Given metric_name {metric_name} is not available in summary"
            )

    def best_value(self, metric_name: str) -> Optional[float]:
        if self.summary_df is None:
            return None
        value = None
        try:
            value = self.summary_df[
                self.summary_df["metric_name"] == metric_name
            ].best_value.values[0]
        except KeyError:
            print(f"Given metric_name {metric_name} is not available in summary")
        finally:
            return value

    def print(self) -> None:
        if self.summary_df is None:
            self.summary_df = pd.DataFrame(self.summary)
        print(self.summary_df)


class ConfigGeneratorManager:
    def __init__(self, config_generator: ConfigGenerator) -> None:
        self.config_generator = config_generator
        self.metric_best_entities_dict = self._get_metric_best_entities_dict()
        self.summary = GeneratorSummary()

    def update(self, loop: Loop, config_idx: int) -> None:
        if self.metric_best_entities_dict is None:
            return None
        self._update_bms(loop, config_idx)
        self._update_bmr(loop, config_idx)

    def finalize(self, logdir: Optional[PATH_STR] = None) -> None:
        if self.metric_best_entities_dict is None:
            return None
        self.summary.make_summary(self.metric_best_entities_dict)
        if logdir is not None:
            self.summary.save_summary(logdir)
        self.summary.print()

    def get_best_config(
        self, summary_path: PATH_STR, metric_name: Optional[str] = None
    ) -> BaseConfig:
        self.summary.load_summary(summary_path)
        if metric_name is None:
            metric_name = self.config_generator.base_config.monitor
        config_idx = self.summary.best_config_idx(metric_name)
        return self.config_generator.get_config(config_idx)

    def _update_bms(self, loop: Loop, config_idx: int) -> None:
        current_metric_setup = get_best_model_metric_setup(loop)
        self.metric_best_entities_dict[current_metric_setup.name].update(
            current_metric_setup, config_idx
        )

    def _update_bmr(self, loop: Loop, config_idx: int) -> None:
        current_metric_setups = get_metric_best_setups(loop)
        for name, current_metric_setup in current_metric_setups.items():
            self.metric_best_entities_dict[name].update(
                current_metric_setup, config_idx
            )

    def _get_metric_best_entities_dict(
        self,
    ) -> Optional[DICT_STR_MBE]:
        train_callbacks = self.config_generator.base_config.train_callbacks(dev=False)
        bms_callback: BestModelSaver = get_callback(train_callbacks, BestModelSaver)
        bmr_callback: BestMetricsReporter = get_callback(
            train_callbacks, BestMetricsReporter
        )
        if bms_callback is None and bmr_callback is None:
            return None

        metric_entities_dict = {}
        if bms_callback is not None:
            metric_entities_dict[
                bms_callback.metric_best_setup.name
            ] = MetricBestEntity(
                bms_callback.metric_best_setup.name, bms_callback.metric_best_setup.mode
            )
        if bmr_callback is not None:
            for (
                name,
                mode,
            ) in bmr_callback.metrics_name_mode.items():
                metric_entities_dict[name] = MetricBestEntity(name, mode)

        return metric_entities_dict


def dict_cross_product(
    input_dict: Dict[str, List[Any]]
) -> Generator[Dict[str, Any], None, None]:
    return (
        dict(zip(input_dict.keys(), values)) for values in product(*input_dict.values())
    )


def load_config_generator(config_generator_path: Path):
    text = config_generator_path.read_text()

    ctx = {}
    exec(text, ctx)

    config_generator = ctx["config_generator"]

    assert isinstance(config_generator, ConfigGenerator)
    return config_generator
