from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Union


@dataclass
class IOImageData:
    name: Enum
    path: "Path"
    ext: Optional[str] = None
    swap_rb_channels: bool = False
    scale: Union[float, Tuple[float, float]] = 1.0

    def __post_init__(self) -> None:
        self.is_dir: bool = self.path.is_dir()
