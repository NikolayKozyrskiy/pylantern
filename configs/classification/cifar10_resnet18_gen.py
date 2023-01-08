from pathlib import Path

from pylantern.config_generator import ConfigGenerator
from pylantern.classification.config import ClassificationConfig


config_generator = ConfigGenerator(
    base_config_path=Path("configs/classification/cifar10_resnet18.py"),
    base_config_class=ClassificationConfig,
    comment_postfix="gen_debug_3",
    variable_parameters={
        "max_epoch": [2],
        "lr": [1.0, 0.05, 0.005],
        "batch_size_train": [100],
        "single_pass_length": [0.25],
    },
)
