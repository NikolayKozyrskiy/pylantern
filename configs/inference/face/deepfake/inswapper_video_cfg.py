from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import torch

from pylantern.common.io.video.data import InputVideoData, OutputVideoData
from pylantern.inference.face.deepfake.data import CropPasteMethod, VideoIOTypes
from pylantern.inference.face.deepfake.df_config import (
    DeepFakeInferenceConfig,
    DFVideoConfig,
)
from pylantern.inference.face.deepfake.models.models import load_inswapper128_onnx
from pylantern.inference.face.deepfake.stages.enhancement import (
    FaceEnhancer,
    load_enhancer_realesrgan,
    load_face_enhancer_codeformer,
    load_face_enhancer_tencent,
)
from pylantern.inference.face.deepfake.stages.face_swap import (
    FaceSwapper,
    load_face_swapper_infa,
)
from pylantern.model_zoo.models import gfpgan_v14

if TYPE_CHECKING:
    from pylantern.inference.face.deepfake.stages.alignment import FaceAlignerInfa


class Cfg(DeepFakeInferenceConfig):
    def face_swapper(
        self,
        face_aligner: Optional["FaceAlignerInfa"],
        device: Union[str, torch.device],
        *args,
        **kwargs
    ) -> Optional["FaceSwapper"]:
        face_swapper_model = load_inswapper128_onnx(
            model_path=Path("_d") / "infa_checkpoints" / "inswapper_128.onnx"
        )
        face_aligner = face_aligner if face_aligner is not None else self.face_aligner()
        return load_face_swapper_infa(
            face_swapper_model=face_swapper_model,
            face_aligner=face_aligner,
            image_size=self.image_size,
            predict_mask=self.predict_mask,
            crop_paste_method=self.crop_paste_method,
            face_segmenter=self.face_segmenter(device=device),
        )

    def face_enhancer(
        self, device: Union[str, torch.device], *args, **kwargs
    ) -> "FaceEnhancer":
        return load_face_enhancer_tencent(
            enhancement_model=gfpgan_v14(root_dir="_d"),
            face_helper=self.face_restore_helper(device=device),
            device=device,
        )


video_config = DFVideoConfig(
    src_img_path=Path("_d/videos/src_imgs/sergei.jpg"),
    src_face_idx=None,
    input_video_data=InputVideoData(
        name=VideoIOTypes.INPUT,
        path=Path("_d/videos/orig/cb_3.mp4"),
    ),
    swapped_video_data=OutputVideoData(
        name=VideoIOTypes.SWAPPED,
        path=Path("_d/videos/swapped/cb_3.mp4"),
    ),
    enhanced_video_data=OutputVideoData(
        name=VideoIOTypes.SWAPPED_ENHANCED,
        path=Path("_d/videos/swapped_enhanced/cb_3.mp4"),
    ),
    concat_video_names=["o", "e"],
    concat_type="a",
)

config = Cfg(
    root_path=Path("_d"),
    stage_names=["swap", "enhance_swapped_dst_img"],
    video_config=video_config,
    predict_mask=False,
    crop_paste_method=CropPasteMethod.INFA_INSWAPPER,
    input_face_size=(512, 512),
    image_size=(512, 512),
    paste_back=True,
    denoise_enhanced=True,
)
