from itertools import product
from pathlib import Path
from typing import Any, Union, Dict, List, Type, Generator

from .config import BaseConfig, load_config


class ConfigGenerator:
    def __init__(
        self,
        base_config_path: Union[Path, str],
        base_config_class: Type[BaseConfig],
        variable_parameters: Dict[str, List[Any]],
    ) -> None:
        self.base_config_path = base_config_path
        self.base_config_class = base_config_class
        self.variable_parameters = variable_parameters
        self._cur_iter_idx = None

    def __call__(self, *args: Any, **kwargs: Any) -> Generator[BaseConfig, None, None]:
        return self.iterate_configs()

    def preprocess(self, *args: Any, **kwargs: Any):
        pass

    def postprocess(self, *args: Any, **kwargs: Any):
        pass

    def iterate_configs(
        self, *args: Any, **kwargs: Any
    ) -> Generator[BaseConfig, None, None]:
        for iter_idx, param_dict in enumerate(
            dict_cross_product(self.variable_parameters)
        ):
            config = self._compile_config(param_dict)
            yield config

    def get_config(self, config_idx: int) -> BaseConfig:
        config_params = list(dict_cross_product(self.variable_parameters))
        return self._compile_config(config_params[config_idx])

    def _compile_config(self, param_dict: Dict[str, Any]) -> BaseConfig:
        config = load_config(self.base_config_path, self.base_config_class)
        for k, v in param_dict.items():
            config.__setattr__(k, v)
        return config


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
