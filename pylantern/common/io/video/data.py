from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import ffmpeg
import imageio_ffmpeg as iffmpeg
import numpy as np
from ffmpeg import Stream

FFMPEG_DIGITS_NUM = 6
FPS_LIST = np.array([24.0, 30.0, 60.0, 120.0, 240.0])


@dataclass
class VideoMeta:
    width: int
    height: int
    duration: float
    frames_num: int
    audio: Optional[Stream] = None

    def __post_init__(self) -> None:
        self.fps: float = FPS_LIST[
            abs(FPS_LIST - self.frames_num / self.duration).argmin()
        ]


@dataclass
class InputVideoData:
    name: Enum
    path: "Path"
    scale: Union[float, Tuple[float, float]] = 1.0
    pix_fmt: str = "bgr24"
    bpp: int = 24
    meta: Optional[VideoMeta] = None

    def __post_init__(self) -> None:
        self.scale = (
            self.scale if isinstance(self.scale, Sequence) else (self.scale, self.scale)
        )
        self.meta = get_video_meta(self.path)
        self.width = int(self.meta.width * self.scale[0]) // 2 * 2
        self.height = int(self.meta.height * self.scale[1]) // 2 * 2
        self.fps = self.meta.fps
        self.bytes_per_pixel = int(self.bpp / 8)

    def __len__(self) -> int:
        return self.meta.frames_num


@dataclass
class OutputVideoData:
    name: Enum
    path: "Path"
    scale: Union[float, Tuple[float, float]] = 1.0
    pix_fmt: str = "yuv420p"
    bpp: int = 24
    vcodec: str = "libx264"
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None

    def __post_init__(self) -> None:
        self.scale = (
            self.scale if isinstance(self.scale, Sequence) else (self.scale, self.scale)
        )
        self.bytes_per_pixel = int(self.bpp / 8)

    def set_resolution(self, width: int, height: int) -> None:
        self.width = int(width * self.scale[0]) // 2 * 2
        self.height = int(height * self.scale[0]) // 2 * 2

    def set_fps(self, fps: float) -> None:
        self.fps = fps


def get_video_meta(video_path: "Path") -> Optional[VideoMeta]:
    try:
        probe = ffmpeg.probe(video_path)
        video_streams = [
            stream for stream in probe["streams"] if stream["codec_type"] == "video"
        ]
        has_audio = any(stream["codec_type"] == "audio" for stream in probe["streams"])
        frames_num, duration = iffmpeg.count_frames_and_secs(video_path)

        return VideoMeta(
            width=int(video_streams[0]["width"]),
            height=int(video_streams[0]["height"]),
            duration=duration,
            frames_num=frames_num,
            audio=ffmpeg.input(video_path).audio if has_audio else None,
        )
    except:
        return None
