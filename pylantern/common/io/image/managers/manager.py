from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from enum import Enum
from multiprocessing.pool import Pool
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Generator, Optional, Sequence, Union

import numpy as np
import tqdm.auto as tqdm

from pylantern.common.utils import mkdir
from pylantern.common.utils.img import load_img, save_img

if TYPE_CHECKING:
    from pylantern.common.io.image.data import IOImageData


class ImageIOManager:
    def __init__(
        self,
        input_image_data: Optional["IOImageData"] = None,
        output_images_data: Sequence[Optional["IOImageData"]] = None,
    ) -> None:
        self.input_image_data = input_image_data
        self._set_ext(img_data=self.input_image_data, ext=input_image_data.ext)

        self.output_images_data: Dict[Enum, "IOImageData"] = {}
        for output_image in output_images_data:
            if output_image is not None:
                self.output_images_data[output_image.name] = output_image
                self._set_ext(
                    img_data=self.output_images_data[output_image.name],
                    ext=output_image.ext,
                )
                if self.output_images_data[output_image.name].path.suffix == "":
                    mkdir(self.output_images_data[output_image.name].path)
                    self.output_images_data[output_image.name].is_dir = True
                else:
                    mkdir(self.output_images_data[output_image.name].path.parent)
                    self.output_images_data[output_image.name].is_dir = False

        self.read_pool: Optional["Pool"] = None
        self.write_executor: Optional["ProcessPoolExecutor"] = None
        self._last_read_img_path: Optional["Path"] = None

    @contextmanager
    def read_ctx(self):
        try:
            with Pool() as self.read_pool:
                yield
        finally:
            self.read_pool: Optional["Pool"] = None

    @contextmanager
    def write_ctx(self):
        try:
            self._last_read_img_path: Optional["Path"] = None
            with ProcessPoolExecutor() as self.write_executor:
                yield
        finally:
            self._last_read_img_path: Optional["Path"] = None
            self.write_executor: Optional["ProcessPoolExecutor"] = None

    def read_images(
        self, images_num: Optional[int] = None, *args, **kwargs
    ) -> Generator["np.ndarray", None, None]:
        try:
            with Pool() as self.read_pool:
                img_paths = sorted(
                    list(
                        self.input_image_data.path.glob(f"*{self.input_image_data.ext}")
                    )
                    if self.input_image_data.is_dir
                    else [self.input_image_data.path]
                )
                imgs = self.read_pool.starmap_async(
                    load_img,
                    (
                        (img_path, self.input_image_data.swap_rb_channels)
                        for img_path in img_paths
                    ),
                )
                for idx, img in tqdm.tqdm(
                    enumerate(imgs.get()),
                    desc=f"{self.input_image_data.name.value} Images",
                    total=len(img_paths),
                ):
                    self._last_read_img_path = img_paths[idx]
                    yield img
        finally:
            self.read_pool: Optional["Pool"] = None

    def write_image(self, image: "np.ndarray", image_name: Enum) -> None:
        if self.output_images_data.get(image_name, None) is not None:
            path = (
                self.output_images_data[image_name].path / self._last_read_img_path.name
                if self.output_images_data[image_name].is_dir
                else self.output_images_data[image_name].path
            )
            swap_rb_channels = self.output_images_data[image_name].swap_rb_channels
            self.write_executor.submit(save_img, image, path, swap_rb_channels)

    def _set_ext(self, img_data: "IOImageData", ext: Optional[str]) -> None:
        if ext is not None:
            img_data.ext = f".{ext}" if not ext.startswith(".") else ext
        else:
            img_data.ext = ""
        return None
