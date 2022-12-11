from pathlib import Path

from pylantern.config_generator import ConfigGenerator
from pylantern.classification.config import ClassificationConfig


config_generator = ConfigGenerator(
    base_config_path=Path("configs/classification/cifar10_resnet18.py"),
    base_config_class=ClassificationConfig,
    variable_parameters={"lr": [1.0, 0.5, 0.001], "batch_size_train": [100, 200]},
)
