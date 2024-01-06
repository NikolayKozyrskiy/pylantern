from pylantern.model_zoo.pix2pix import MultiscaleDiscriminator, p2p_discriminator

from .config import GanPix2PixConfig


class Pix2PixHDConfig(GanPix2PixConfig):
    # '# of input image channels'
    input_nc: int = 3
    # 'weight for feature matching loss'
    lambda_feat: float = 10.0
    n_blocks_global: int = 9
    # 'number of residual blocks in the local enhancer network'
    n_blocks_local: int = 3
    n_clusters: int = 10
    n_downsample_E: int = 4
    # 'number of downsampling layers in netG'
    n_downsample_global: int = 4
    n_layers_D: int = 3
    # 'number of local enhancers to use'
    n_local_enhancers: int = 1
    ndf: int = 64
    nef: int = 16
    # 'selects model to use for netG: global, local, u2net'
    netG: str = "global"
    # '# of gen filters in first conv layer'
    ngf: int = 64
    # '# of iter at starting learning rate'
    niter: int = 100
    # '# of iter to linearly decay learning rate to zero'
    niter_decay: int = 100
    # 'number of epochs that we only train the outmost local enhancer'
    niter_fix_global: int = 0
    no_ganFeat_loss: bool = False
    no_instance: bool = True
    no_lsgan: bool = False
    norm: str = "instance"
    num_D: int = 2
    output_nc: int = 3

    def discriminator_model(self, *args, **kwargs) -> "MultiscaleDiscriminator":
        return p2p_discriminator(
            self.input_nc + self.output_nc,
            ndf=self.ndf,
            n_layers_D=self.n_layers_D,
            norm=self.norm,
            use_sigmoid=self.no_lsgan,
            num_D=self.num_D,
            getIntermFeat=not self.no_ganFeat_loss,
        )
