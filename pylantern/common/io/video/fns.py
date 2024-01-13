import os
import subprocess
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple, Union

import ffmpeg
import imageio_ffmpeg as iffmpeg
import numpy as np
from ffmpeg import Stream

from pylantern.common.constants import FFMPEG_BIN
from pylantern.common.utils import mkdir

# Look here for inspiration:
# https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan_video.py

FFMPEG_DIGITS_NUM = 6
FPS_LIST = np.array([24.0, 30.0, 60.0, 120.0, 240.0])


class VideoMeta(NamedTuple):
    width: int
    height: int
    duration: float
    frames_num: int
    audio: Optional["Stream"] = None

    @property
    def fps(self) -> float:
        fps = self.frames_num / self.duration
        return FPS_LIST[abs(FPS_LIST - fps).argmin()]


def get_video_meta(video_path: "Path") -> "VideoMeta":
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


def get_fps(video_path: "Path") -> float:
    # return cv2.VideoCapture(str(video_path)).get(cv2.CAP_PROP_FPS)
    frames, secs = iffmpeg.count_frames_and_secs(video_path)
    fps = frames / secs
    return FPS_LIST[abs(FPS_LIST - fps).argmin()]


def get_sub_video(video_path: "Path", num_process: int, process_idx: int) -> "Path":
    if num_process == 1:
        return video_path
    meta = get_video_meta(video_path=video_path)
    part_time = meta.duration // num_process
    out_path = mkdir(video_path.parent / f"{video_path.stem}_tmp_chunks")
    out_path = out_path / f"{process_idx:03d}.mp4"
    cmd = [
        FFMPEG_BIN,
        f"-i {video_path}",
        "-ss",
        f"{part_time * process_idx}",
        f"-to {part_time * (process_idx + 1)}"
        if process_idx != num_process - 1
        else "",
        "-async 1",
        out_path,
        "-y",
    ]

    subprocess.call(" ".join(cmd), shell=True)
    return out_path


def imgs2video(
    src_imgs_path: Path,
    video_path: Optional[Path] = None,
    fps: float = 30.0,
    img_ext: str = "png",
) -> Path:
    # TODO: add different video extensions and corresponding codecs
    video_path = (
        src_imgs_path.parent / f"{src_imgs_path.name}.mp4"
        if video_path is None
        else video_path.parent / f"{video_path.stem}.mp4"
    )
    cmd = (
        f"ffmpeg -y -framerate {fps} -pattern_type glob -i '{src_imgs_path}/*.{img_ext}' "
        f'-c:v libx264 -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
        f"-pix_fmt yuv420p {video_path}"
    )
    os.system(cmd)
    return video_path


def videos2imgs(
    root_src_path: Path,
    video_names: List[Path],
    root_dst_path: Optional[Path] = None,
    img_ext: str = "png",
    fps: Optional[float] = None,
    scale: str = "0:0",
) -> None:
    for video_name in video_names:
        video_path = root_src_path / video_name
        _fps = get_fps(video_path) if fps is None else fps
        vf = f'-vf "scale={scale},fps={_fps}"' if scale != "0:0" else f"-vf fps={_fps}"
        dst_path = mkdir(
            root_src_path / video_name.stem if root_dst_path is None else root_dst_path
        )
        cmd = f"ffmpeg -y -i {video_path} {vf} {dst_path}/%0{FFMPEG_DIGITS_NUM}d.{img_ext}"
        os.system(cmd)
    return None


def hstack_videos(
    video_paths: List[Path],
    result_path: Path,
) -> None:
    num = len(video_paths)
    assert num > 1, f"Number of videos must be > 1, found {num}"
    mkdir(result_path.parent)

    cmd = "ffmpeg "
    add_vsync = False
    for i, vn in enumerate(video_paths):
        cmd += f"-i {vn} "
        if vn.suffix in [".png", ".jpg"] or add_vsync:
            add_vsync = True
    if add_vsync:
        cmd += f"-vsync 2 "
    cmd += f"-filter_complex hstack=inputs={num} {result_path}"
    os.system(cmd)
    return None


def vstack_videos(
    video_paths: List[Path],
    result_path: Path,
) -> None:
    num = len(video_paths)
    assert num > 1, f"Number of videos must be > 1, found {num}"
    mkdir(result_path.parent)

    cmd = "ffmpeg "
    for i, vn in enumerate(video_paths):
        cmd += f"-i {vn} "

    cmd += f"-filter_complex vstack=inputs={num} -vsync 2 {result_path}"
    os.system(cmd)
    return None


def downscale_videos(
    root_dir: Path,
    video_names: List[Path],
    scale: int,
    new_width: int,
    new_height: int,
) -> None:
    for video_name in video_names:
        src_video_path = root_dir / video_name
        if scale > -1.0:
            scale_params = f'-vf "scale=iw/{scale}:ih/{scale}"'
            dst_video_path = (
                root_dir / f"{video_name.stem}_resized_x{scale}{video_name.suffix}"
            )
        else:
            scale_params = f"-vf scale={new_width}:{new_height}"
            dst_video_path = (
                root_dir
                / f"{video_name.stem}_resized_{new_width}_{new_height}{video_name.suffix}"
            )
        cmd = f"ffmpeg -i {src_video_path} {scale_params} {dst_video_path}"
        os.system(cmd)
    return None


def upscale_videos(
    root_dir: Path,
    video_names: List[Path],
    scale: int,
    new_width: int,
    new_height: int,
) -> None:
    for video_name in video_names:
        src_video_path = root_dir / video_name
        dst_video_path = root_dir / f"{video_name.stem}_resized.{video_name.suffix}"
        if scale > -1.0:
            scale_params = f'-vf "scale=iw*{scale}:ih*{scale}"'
        else:
            scale_params = f"-vf scale={new_width}:{new_height}"
        cmd = f"ffmpeg -i {src_video_path} {scale_params} {dst_video_path}"
        os.system(cmd)
    return None


def video_scale_to_ffmpeg(video_scale: Union[float, Tuple[float, float]]) -> str:
    video_scale_ffmpeg = "0:0"
    if isinstance(video_scale, float):
        if video_scale == 1.0:
            video_scale_ffmpeg = "0:0"
        elif video_scale < 1.0:
            scale_int = int(round(1 / video_scale))
            video_scale_ffmpeg = f"iw/{scale_int}:ih/{scale_int}"
        else:
            scale_int = int(round(video_scale))
            video_scale_ffmpeg = f"iw*{scale_int}:ih*{scale_int}"
    else:
        s_w, s_h = video_scale[0], video_scale[1]
        if s_w == 1.0:
            video_scale_ffmpeg = "0:"
        elif s_w < 1.0:
            scale_int = int(round(1 / s_w))
            video_scale_ffmpeg = f"iw/{scale_int}:"
        else:
            scale_int = int(round(s_w))
            video_scale_ffmpeg = f"iw*{scale_int}:"

        if s_h == 1.0:
            video_scale_ffmpeg += "0"
        elif s_h < 1.0:
            scale_int = int(round(1 / s_h))
            video_scale_ffmpeg += f"ih/{scale_int}"
        else:
            scale_int = int(round(s_h))
            video_scale_ffmpeg += f"ih*{scale_int}"
    return video_scale_ffmpeg
