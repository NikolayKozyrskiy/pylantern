import re

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from torch import Tensor

# from .sync_batchnorm import SynchronizedBatchNorm2d
from torch.nn import SyncBatchNorm as SynchronizedBatchNorm2d
from torch.nn import init


def spade_generator(
    input_nc: int,
    output_nc: int,
    image_size: int,
    aspect_ratio: float = 1,
    ngf: int = 64,
    num_upsampling_layers: str = "normal",
    norm_G: str = "spectralspadesyncbatch3x3",
    init_type: str = "xavier",
    init_variance: float = 0.02,
) -> nn.Module:
    net = SPADEGenerator(
        input_nc,
        output_nc,
        ngf,
        num_upsampling_layers,
        image_size,
        aspect_ratio,
        norm_G,
    )
    net.init_weights(init_type, init_variance)
    return net


class SPADE(nn.Module):
    def __init__(self, norm_G, norm_nc, label_nc, interpolation_mode="bilinear"):
        super().__init__()
        self.interpolation_mode = interpolation_mode
        assert norm_G.startswith("spade")
        parsed = re.search("spade(\D+)(\d)x\d", norm_G)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        # param_free_norm_type = 'syncbatch'
        # ks = 3

        if param_free_norm_type == "instance":
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == "syncbatch":
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == "batch":
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError(
                "%s is not a recognized param-free norm type in SPADE"
                % param_free_norm_type
            )

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw), nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode=self.interpolation_mode)
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, input_nc, norm_G):
        super().__init__()
        # Attributes
        self.learned_shortcut = fin != fout
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if "spectral" in norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        norm_G = norm_G.replace("spectral", "")
        self.norm_0 = SPADE(norm_G, fin, input_nc)
        self.norm_1 = SPADE(norm_G, fmiddle, input_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(norm_G, fin, input_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(
            "Network [%s] was created. Total number of parameters: %.1f million. "
            "To see the architecture, do print(network)."
            % (type(self).__name__, num_params / 1000000)
        )

    def init_weights(self, init_type="normal", gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find("BatchNorm2d") != -1:
                if hasattr(m, "weight") and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, "bias") and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, "weight") and (
                classname.find("Conv") != -1 or classname.find("Linear") != -1
            ):
                if init_type == "normal":
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == "xavier":
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == "xavier_uniform":
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == "kaiming":
                    init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                elif init_type == "orthogonal":
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == "none":  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        "initialization method [%s] is not implemented" % init_type
                    )
                if hasattr(m, "bias") and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, "init_weights"):
                m.init_weights(init_type, gain)


class SPADEGenerator(BaseNetwork):
    def __init__(
        self,
        input_nc,
        output_nc,
        ngf,
        num_upsampling_layers,
        crop_size,
        aspect_ratio,
        norm_G,
    ):
        super().__init__()
        nf = ngf
        self.num_upsampling_layers = num_upsampling_layers
        self.output_nc = output_nc
        self.sw, self.sh, self.num_up_layers = self.compute_latent_vector_size(
            crop_size, aspect_ratio
        )

        # Otherwise, we make the network deterministic by starting with
        # downsampled segmentation map instead of random z
        self.fc = nn.Conv2d(input_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, input_nc, norm_G)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, input_nc, norm_G)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, input_nc, norm_G)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, input_nc, norm_G)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, input_nc, norm_G)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, input_nc, norm_G)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, input_nc, norm_G)

        final_nc = nf

        if num_upsampling_layers == "most":
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, input_nc, norm_G)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, output_nc, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, crop_size, aspect_ratio):
        if self.num_upsampling_layers == "normal":
            num_up_layers = 5
        elif self.num_upsampling_layers == "more":
            num_up_layers = 6
        elif self.num_upsampling_layers == "most":
            num_up_layers = 7
        else:
            raise ValueError(
                "num_upsampling_layers [%s] not recognized" % self.num_upsampling_layers
            )

        sw = crop_size // (2**num_up_layers)
        sh = round(sw / aspect_ratio)

        return sw, sh, num_up_layers

    def forward(self, input_image: Tensor) -> Tensor:
        seg = input_image

        # we downsample segmap and run convolution
        # if ignore_crop_size:
        sw = input_image.shape[2] // (2**self.num_up_layers)
        sh = round(sw / (input_image.shape[2] / input_image.shape[3]))
        # else:
        #     sh, sw = self.sh, self.sw
        x = F.interpolate(seg, size=(sh, sw))
        x = self.fc(x)

        x = self.head_0(x, seg)

        x = self.up(x)
        x = self.G_middle_0(x, seg)

        if self.num_upsampling_layers == "more" or self.num_upsampling_layers == "most":
            x = self.up(x)

        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        if self.num_upsampling_layers == "most":
            x = self.up(x)
            x = self.up_4(x, seg)

        if self.output_nc <= 3:
            x = self.conv_img(F.leaky_relu(x, 2e-1))
            x = F.tanh(x)
        else:
            x = self.conv_img(x)

        return x

    def predict(self, input_image: Tensor) -> Tensor:
        return self.forward(input_image=input_image)
