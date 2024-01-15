from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from insightface.app import FaceAnalysis

from pylantern.common.constants import DEFAULT_IMG_MEAN, DEFAULT_IMG_STD
from pylantern.common.io.video.data import InputVideoData, OutputVideoData
from pylantern.inference.config import BaseInferenceConfig
from pylantern.inference.face.deepfake.data import CropPasteMethod
from pylantern.inference.face.deepfake.models.models import (
    load_generator_inference_model,
    load_inswapper128_onnx,
)
from pylantern.inference.face.deepfake.stages.alignment import (
    FaceAlignerInfa,
    load_face_aligner_infa,
)
from pylantern.inference.face.deepfake.stages.enhancement import (
    FaceEnhancer,
    load_enhancer_realesrgan,
    load_face_enhancer_codeformer,
    load_face_enhancer_tencent,
)
from pylantern.inference.face.deepfake.stages.face_swap import (
    FaceSwapper,
    load_face_swapper_generator,
    load_face_swapper_infa,
)
from pylantern.inference.face.deepfake.stages.segmentation import (
    FaceSegmenter,
    load_face_segmenter_parsenet,
)
from pylantern.model_zoo.gfpgan import gfpgan_generator
from pylantern.model_zoo.models import gfpgan_v14

if TYPE_CHECKING:
    pass


class DFImageConfig(NamedTuple):
    dst_img_path: Optional["Path"] = None  # Path("imgs__orig/img.png")
    dst_imgs_dir: Optional["Path"] = None  # Path("imgs__orig")
    dst_imgs_aligned_dir: Optional["Path"] = None
    swapped_imgs_dir: Optional["Path"] = None
    enhanced_imgs_dir: Optional["Path"] = None


# class DFVideoConfig(NamedTuple):
#     dst_video_path: Optional["Path"] = None  # Path("videos__orig/video__orig.mp4")
#     dst_video_aligned_path: Optional["Path"] = None
#     swapped_video_path: Optional["Path"] = None
#     enhanced_video_path: Optional["Path"] = None
#     dst_videos_dir: Optional["Path"] = None  # Path("videos__orig")
#     swapped_videos_dir: Optional["Path"] = None
#     enhanced_videos_dir: Optional["Path"] = None
#     concat_video_names: Optional[List[str]] = None
#     concat_type: str = "h"


class DFVideoConfig(NamedTuple):
    src_img_path: "Path"
    src_face_idx: Optional[int] = None
    input_video_data: Optional["InputVideoData"] = None
    swapped_video_data: Optional["OutputVideoData"] = None
    enhanced_video_data: Optional["OutputVideoData"] = None
    concat_video_names: Optional[List[str]] = None
    concat_type: str = "h"


class DeepFakeInferenceConfig(BaseInferenceConfig):
    image_config: Optional["DFImageConfig"] = None
    video_config: Optional["DFVideoConfig"] = None

    input_face_size: Tuple[int, int] = (256, 256)
    image_size: Tuple[int, int] = (512, 512)
    upscale_coeff: int = 1

    paste_back: bool = True
    denoise_enhanced: bool = True
    predict_mask: bool = True
    crop_paste_method: CropPasteMethod = CropPasteMethod.INFA_INSWAPPER

    def face_aligner(self) -> "FaceAlignerInfa":
        return load_face_aligner_infa(
            input_face_size=self.input_face_size,
            face_analyzer=self.face_analyzer_infa(),
        )

    def face_analyzer_infa(self) -> "FaceAnalysis":
        """
        root_path: default = checkpoints
        """
        face_analyser = FaceAnalysis(
            name="buffalo_l", root=self.root_path.parent / "infa_checkpoints"
        )
        face_analyser.prepare(ctx_id=0, det_size=(320, 320))
        return face_analyser

    def face_segmenter(
        self, device: Union[str, torch.device], *args, **kwargs
    ) -> Optional["FaceSegmenter"]:
        return load_face_segmenter_parsenet(
            weights_path=Path("_d/gfpgan/weights"), device=device
        )

    def face_swapper(
        self,
        face_aligner: Optional["FaceAlignerInfa"],
        device: Union[str, torch.device],
        *args,
        **kwargs
    ) -> Optional["FaceSwapper"]:
        face_swapper_model = load_generator_inference_model(
            generator_model=gfpgan_generator(
                arch="orig",
                predict_mask=self.predict_mask,
                out_size=self.image_size[0],
                decoder_load_path=None,
                fix_decoder=False,
                num_style_feat=self.image_size[0],
                channel_multiplier=1,
                resample_kernel=(1, 3, 3, 1),
                num_mlp=8,
                lr_mlp=0.01,
                input_is_latent=True,
                different_w=True,
                narrow=1.0,
                sft_half=True,
            ),
            mean=DEFAULT_IMG_MEAN,
            std=DEFAULT_IMG_STD,
            checkpoint_path=self.checkpoint_path,
            device=device,
        )
        face_aligner = face_aligner if face_aligner is not None else self.face_aligner()
        return load_face_swapper_generator(
            face_swapper_model=face_swapper_model,
            face_aligner=face_aligner,
            image_size=self.image_size,
            predict_mask=self.predict_mask,
            crop_paste_method=self.crop_paste_method,
            device=device,
        )

    def face_enhancer(
        self, device: Union[str, torch.device], *args, **kwargs
    ) -> "FaceEnhancer":
        return load_face_enhancer_tencent(
            enhancement_model=gfpgan_v14(root_dir="_d"),
            face_helper=self.face_restore_helper(device=device),
            device=device,
        )

    def face_restore_helper(
        self, device: Union[str, torch.device]
    ) -> "FaceRestoreHelper":
        face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=device,
            model_rootpath="_d/gfpgan/weights",
        )
        return face_helper
