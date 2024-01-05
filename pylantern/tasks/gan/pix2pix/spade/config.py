from ..pix2pixhd import Pix2PixHDConfig


class SpadeConfig(Pix2PixHDConfig):
    # choices=('normal', 'more', 'most')
    # "If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")
    num_upsampling_layers: str = "normal"
    # network initialization [normal|xavier|kaiming|orthogonal]
    init_type: str = "xavier"
    aspect_ratio: float = 1.0
    norm_G: str = "spectralspadesyncbatch3x3"
    init_variance: float = 0.02
