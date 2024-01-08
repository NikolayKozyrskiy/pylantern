from .face import (
    FaceGFPGANPipeline,
    FacePix2PixHDPipeline,
    FacePix2PixPipeline,
    FaceSpadePipeline,
    face_gfpgan_pipeline_from_config,
    face_pix2pixhd_pipeline_from_config,
    face_spade_pipeline_from_config,
)
from .general import (
    GFPGANPipeline,
    Pix2PixHDPipeline,
    gfpgan_pipeline_from_config,
    pix2pixhd_pipeline_from_config,
)
from .pipeline import BasePix2PixPipeline, base_pix2pix_pipeline_from_config
