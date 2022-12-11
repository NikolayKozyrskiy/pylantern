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

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        for param_dict in dict_cross_product(self.variable_parameters):
            config = load_config(self.base_config_path, self.base_config_class)
            for k, v in param_dict.items():
                config.__setattr__(k, v)
            yield config


def dict_cross_product(input_dict: Dict[str, List[Any]]) -> Generator[Dict[str, Any]]:
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
