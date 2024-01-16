from typing import TYPE_CHECKING, Optional

from matches.utils import seed_everything, setup_cudnn_reproducibility

from pylantern.common.io.image.managers import ImageIOManager
from pylantern.common.io.video.managers import VideoIOManager
from pylantern.common.utils import get_device
from pylantern.config import load_config
from pylantern.inference.face.deepfake.data import FaceSwapData, IOTypes
from pylantern.inference.face.deepfake.df_config import DeepFakeInferenceConfig
from pylantern.inference.face.deepfake.df_output_dispatcher import (
    DeepFakeInferenceOutputDispatcher,
)
from pylantern.inference.face.deepfake.df_pipeline import (
    deepfake_inference_pipeline_from_config,
)

if TYPE_CHECKING:
    from pathlib import Path


def infer_video(config_path: "Path", frames_num: Optional[int] = None) -> None:
    config: "DeepFakeInferenceConfig" = load_config(
        config_path=config_path, desired_class=DeepFakeInferenceConfig
    )
    seed_everything(42)
    setup_cudnn_reproducibility(deterministic=False, benchmark=True)
    device = get_device()

    pipeline = deepfake_inference_pipeline_from_config(config=config, device=device)
    out_dispatcher = DeepFakeInferenceOutputDispatcher(stage_names=config.stage_names)
    io_manager = VideoIOManager(
        input_video=config.video_config.input_video_data,
        output_videos=(
            config.video_config.swapped_video_data,
            config.video_config.swapped_enhanced_video_data,
            config.video_config.enhanced_video_data,
        ),
    )

    with pipeline.src_img_scope(
        src_img=config.video_config.src_img_path,
        face_idx=config.video_config.src_face_idx,
    ):
        with io_manager.write_ctx():
            for dst_img in io_manager.read_frames(frames_num=frames_num):
                swap_data = FaceSwapData(dst_img=dst_img)
                with pipeline.data_item_scope(data=swap_data), pipeline.cache_scope():
                    out_dispatcher.compute_stages(pipeline=pipeline)
                    io_manager.write_frame(
                        frame=swap_data.swapped_dst_img,
                        video_name=IOTypes.SWAPPED,
                    )
                    io_manager.write_frame(
                        frame=swap_data.enhanced_swapped_dst_img,
                        video_name=IOTypes.SWAPPED_ENHANCED,
                    )
                    io_manager.write_frame(
                        frame=swap_data.enhanced_dst_img,
                        video_name=IOTypes.ENHANCED,
                    )
    return None


def infer_image(config_path: "Path", images_num: Optional[int] = None) -> None:
    config: "DeepFakeInferenceConfig" = load_config(
        config_path=config_path, desired_class=DeepFakeInferenceConfig
    )
    seed_everything(42)
    setup_cudnn_reproducibility(deterministic=False, benchmark=True)
    device = get_device()

    pipeline = deepfake_inference_pipeline_from_config(config=config, device=device)
    out_dispatcher = DeepFakeInferenceOutputDispatcher(stage_names=config.stage_names)
    io_manager = ImageIOManager(
        input_image_data=config.image_config.dst_img_data,
        output_images_data=(
            config.image_config.swapped_img_data,
            config.image_config.swapped_enhanced_img_data,
            config.image_config.enhanced_dst_img_data,
        ),
    )

    with pipeline.src_img_scope(
        src_img=config.image_config.src_img_path,
        face_idx=config.image_config.src_face_idx,
    ):
        with io_manager.write_ctx():
            for dst_img in io_manager.read_images(images_num=images_num):
                swap_data = FaceSwapData(dst_img=dst_img)
                with pipeline.data_item_scope(data=swap_data), pipeline.cache_scope():
                    out_dispatcher.compute_stages(pipeline=pipeline)
                    io_manager.write_image(
                        image=swap_data.swapped_dst_img,
                        image_name=IOTypes.SWAPPED,
                    )
                    io_manager.write_image(
                        image=swap_data.enhanced_swapped_dst_img,
                        image_name=IOTypes.SWAPPED_ENHANCED,
                    )
                    io_manager.write_image(
                        image=swap_data.enhanced_dst_img,
                        image_name=IOTypes.ENHANCED,
                    )
    return None
