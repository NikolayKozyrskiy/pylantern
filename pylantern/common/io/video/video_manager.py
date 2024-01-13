from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Generator, Optional, Sequence, Tuple, Union

import ffmpeg
import numpy as np

from pylantern.common.constants import FFMPEG_BIN
from pylantern.common.io.video.fns import get_video_meta

if TYPE_CHECKING:
    from subprocess import Popen


class VideoIOManager:
    def __init__(
        self,
        input_video_path: "Path",
        output_video_path: Optional["Path"] = None,
        input_scale: Union[float, Tuple[float, float]] = 1.0,
        output_scale: Union[float, Tuple[float, float]] = 1.0,
        input_pix_fmt: str = "bgr24",
        output_pix_fmt: str = "yuv420p",
        output_vcodec: str = "libx264",
        bpp: int = 24,
    ) -> None:
        self._input_video_path = input_video_path
        self._output_video_path = output_video_path
        self._input_scale = (
            input_scale
            if isinstance(input_scale, Sequence)
            else (input_scale, input_scale)
        )
        self._output_scale = (
            output_scale
            if isinstance(output_scale, Sequence)
            else (output_scale, output_scale)
        )
        self._input_pix_fmt = input_pix_fmt
        self._output_pix_fmt = output_pix_fmt
        self._output_vcodec = output_vcodec
        self._bytes_per_pixel = int(bpp / 8)
        self._input_video_meta = get_video_meta(video_path=input_video_path)
        self._input_video_stream_reader: Optional["Popen"] = None
        self._output_video_stream_writer: Optional["Popen"] = None

    @property
    def input_width(self) -> int:
        return int(self._input_video_meta.width * self._input_scale[0]) // 2 * 2

    @property
    def input_height(self) -> int:
        return int(self._input_video_meta.height * self._input_scale[1]) // 2 * 2

    @property
    def output_width(self) -> int:
        return int(self._input_video_meta.width * self._output_scale[0]) // 2 * 2

    @property
    def output_height(self) -> int:
        return int(self._input_video_meta.height * self._output_scale[1]) // 2 * 2

    @property
    def fps(self) -> float:
        return self._input_video_meta.fps

    def __len__(self):
        return self._input_video_meta.frames_num

    @contextmanager
    def video_scope(self):
        try:
            self._init_streams()
            yield
        finally:
            self._close_streams()

    @contextmanager
    def write_scope(self) -> None:
        try:
            self._open_output_video_stream_writer()
            yield
        finally:
            self._close_output_video_stream_writer()

    @contextmanager
    def read_scope(self) -> None:
        try:
            self._open_input_video_stream_reader()
            yield
        finally:
            self._close_input_video_stream_reader()

    def read_frames(
        self, frames_num: Optional[int] = None, *args, **kwargs
    ) -> Generator["np.ndarray", None, None]:
        try:
            self._open_input_video_stream_reader()
            frames_num = (
                frames_num
                if frames_num is not None
                else self._input_video_meta.frames_num
            )
            for frame_idx in range(frames_num):
                frame = self.read_frame()
                if frame is not None:
                    yield frame
                else:
                    break
        finally:
            self._close_input_video_stream_reader()

    def write_frame(self, frame: "np.ndarray") -> None:
        self._output_video_stream_writer.stdin.write(frame.astype(np.uint8).tobytes())
        return None

    def read_frame(self) -> Optional["np.ndarray"]:
        img_bytes = self._input_video_stream_reader.stdout.read(
            self.input_height * self.input_width * self._bytes_per_pixel
        )
        if not img_bytes:
            return None
        img = np.frombuffer(img_bytes, np.uint8).reshape(
            self.input_height,
            self.input_width,
            3,
        )
        return img

    def _init_streams(self) -> None:
        self._open_input_video_stream_reader()
        self._open_output_video_stream_writer()

    def _open_input_video_stream_reader(self) -> None:
        assert self._input_video_path is not None, "Input video path must be defined!"
        if self._input_video_stream_reader is None:
            self._input_video_stream_reader: "Popen" = (
                ffmpeg.input(self._input_video_path)
                .output(
                    "pipe:",
                    format="rawvideo",
                    pix_fmt=self._input_pix_fmt,
                    s=f"{self.input_width}x{self.input_height}",
                    vf=f"fps={self._input_video_meta.fps}",
                    loglevel="error",
                )
                .run_async(
                    pipe_stdout=True,
                    cmd=FFMPEG_BIN,
                )
            )

    def _open_output_video_stream_writer(self) -> None:
        assert self._output_video_path is not None, "Output video path must be defined!"
        if self._output_video_stream_writer is None:
            stream = ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt=self._input_pix_fmt,
                s=f"{self.input_width}x{self.input_height}",
                framerate=self._input_video_meta.fps,
            )
            if self._input_video_meta.audio is not None:
                stream = ffmpeg.output(
                    stream,
                    self._input_video_meta.audio,
                    filename=self._output_video_path,
                    s=f"{self.output_width}x{self.output_height}",
                    pix_fmt=self._output_pix_fmt,
                    vcodec=self._output_vcodec,
                    loglevel="error",
                    acodec="copy",
                )
            else:
                stream = ffmpeg.output(
                    stream,
                    filename=self._output_video_path,
                    s=f"{self.output_width}x{self.output_height}",
                    pix_fmt=self._output_pix_fmt,
                    vcodec=self._output_vcodec,
                    loglevel="error",
                )
            self._output_video_stream_writer: "Popen" = (
                stream.overwrite_output().run_async(
                    pipe_stdin=True,
                    cmd=FFMPEG_BIN,
                )
            )

    def _close_input_video_stream_reader(self) -> None:
        if self._input_video_stream_reader is not None:
            self._input_video_stream_reader.kill()
            # self._input_video_stream_reader.wait()
            self._input_video_stream_reader = None

    def _close_output_video_stream_writer(self) -> None:
        if self._output_video_stream_writer is not None:
            self._output_video_stream_writer.stdin.close()
            self._output_video_stream_writer.wait()
            self._output_video_stream_writer = None

    def _close_streams(self) -> None:
        self._close_input_video_stream_reader()
        self._close_output_video_stream_writer()
