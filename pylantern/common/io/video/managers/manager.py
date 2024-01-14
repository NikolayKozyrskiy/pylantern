from contextlib import contextmanager
from typing import TYPE_CHECKING, Dict, Generator, Optional, Sequence

import ffmpeg
import numpy as np

from pylantern.common.constants import FFMPEG_BIN
from pylantern.common.io.video.data import InputVideoData, OutputVideoData

if TYPE_CHECKING:
    from subprocess import Popen


class VideoIOManager:
    def __init__(
        self,
        input_video: "InputVideoData",
        output_videos: Sequence["OutputVideoData"],
    ) -> None:
        self.input_video = input_video
        self.read_stream: Optional["Popen"] = None
        self.output_videos: Dict[str, "OutputVideoData"] = {
            video.name: video for video in output_videos
        }
        for video_name in self.output_videos.keys():
            self.output_videos[video_name].set_resolution(
                width=self.input_video.meta.width, height=self.input_video.meta.height
            )
            self.output_videos[video_name].set_fps(fps=self.input_video.fps)
        self.write_streams: Dict[str, Optional["Popen"]] = {
            video.name: None for video in output_videos
        }

    @contextmanager
    def videos_scope(self):
        try:
            self._open_streams()
            yield
        finally:
            self._close_streams()

    @contextmanager
    def write_scope(self, video_name: str) -> None:
        try:
            self._open_output_video_write_stream(video_name=video_name)
            yield
        finally:
            self._close_output_video_write_stream(video_name=video_name)

    @contextmanager
    def read_scope(self) -> None:
        try:
            self._open_input_video_read_stream()
            yield
        finally:
            self._close_input_video_read_stream()

    def read_frames(
        self, frames_num: Optional[int] = None, *args, **kwargs
    ) -> Generator["np.ndarray", None, None]:
        try:
            self._open_input_video_read_stream()
            frames_num = (
                frames_num
                if frames_num is not None
                else self.input_video.meta.frames_num
            )
            for frame_idx in range(frames_num):
                frame = self.read_frame()
                if frame is not None:
                    yield frame
                else:
                    break
        finally:
            self._close_input_video_read_stream()

    def write_frame(self, frame: "np.ndarray", video_name: str) -> None:
        self.write_streams[video_name].stdin.write(frame.astype(np.uint8).tobytes())
        return None

    def read_frame(self) -> Optional["np.ndarray"]:
        img_bytes = self.read_stream.stdout.read(
            self.input_video.height
            * self.input_video.width
            * self.input_video.bytes_per_pixel
        )
        if not img_bytes:
            return None
        img = np.frombuffer(img_bytes, np.uint8).reshape(
            self.input_video.height,
            self.input_video.width,
            3,
        )
        return img

    def _open_input_video_read_stream(self) -> None:
        if self.read_stream is None:
            self.read_stream: "Popen" = (
                ffmpeg.input(self.input_video.path)
                .output(
                    "pipe:",
                    format="rawvideo",
                    pix_fmt=self.input_video.pix_fmt,
                    s=f"{self.input_video.width}x{self.input_video.height}",
                    vf=f"fps={self.input_video.fps}",
                    loglevel="error",
                )
                .run_async(
                    pipe_stdout=True,
                    cmd=FFMPEG_BIN,
                )
            )

    def _open_output_video_write_stream(self, video_name: str) -> None:
        if self.write_streams[video_name] is None:
            stream = ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt=self.input_video.pix_fmt,
                s=f"{self.input_video.width}x{self.input_video.height}",
                framerate=self.input_video.fps,
            )
            if self.input_video.meta.audio is not None:
                stream = ffmpeg.output(
                    stream,
                    self.input_video.meta.audio,
                    filename=self.output_videos[video_name].path,
                    s=f"{self.output_videos[video_name].width}x{self.output_videos[video_name].height}",
                    pix_fmt=self.output_videos[video_name].pix_fmt,
                    vcodec=self.output_videos[video_name].vcodec,
                    loglevel="error",
                    acodec="copy",
                )
            else:
                stream = ffmpeg.output(
                    stream,
                    filename=self.output_videos[video_name].path,
                    s=f"{self.output_videos[video_name].width}x{self.output_videos[video_name].height}",
                    pix_fmt=self.output_videos[video_name].pix_fmt,
                    vcodec=self.output_videos[video_name].vcodec,
                    loglevel="error",
                )
            self.write_streams[
                video_name
            ]: "Popen" = stream.overwrite_output().run_async(
                pipe_stdin=True,
                cmd=FFMPEG_BIN,
            )

    def _open_streams(self) -> None:
        self._open_input_video_read_stream()
        for video_name in self.write_streams.keys():
            self._open_output_video_write_stream(video_name=video_name)

    def _close_input_video_read_stream(self) -> None:
        if self.read_stream is not None:
            self.read_stream.kill()
            # self.read_stream.wait()
            self.read_stream = None

    def _close_output_video_write_stream(self, video_name: str) -> None:
        if self.write_streams[video_name] is not None:
            self.write_streams[video_name].stdin.close()
            self.write_streams[video_name].wait()
            self.write_streams[video_name] = None

    def _close_streams(self) -> None:
        self._close_input_video_read_stream()
        for video_name in self.write_streams.keys():
            self._close_output_video_write_stream(video_name=video_name)
