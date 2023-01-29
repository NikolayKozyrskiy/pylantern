import torch
import torch.nn as nn
import torch.nn.functional as F

from pylantern.dnn_compression.quantization import QActivation


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, num_bits: int, block_name, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.num_bits = num_bits
        self.block_name = block_name
        self.qact_conv1 = QActivation(
            num_bits, name=f"QActivation__{self.block_name}_qact_conv1", is_active=True
        )
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.qact_conv2 = QActivation(
            num_bits, name=f"QActivation__{self.block_name}_qact_conv2", is_active=True
        )
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.qact_conv1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.qact_conv2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        #         out = F.relu1(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        #         out = F.relu1(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_bits: int, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.num_bits = num_bits
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(1, block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(2, block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(3, block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(4, block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block_num, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            block_str = f"block_{block_num}_{stride}_"
            layers.append(
                block(self.num_bits, block_str, self.in_planes, planes, stride)
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, out_feature=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.layer1(out))
        out = F.relu(self.layer2(out))
        out = F.relu(self.layer3(out))
        out = self.layer4(out)
        #         feature = out.view(out.size(0), -1)
        out = F.relu(out)
        out = F.avg_pool2d(out, 4)
        feature = out.view(out.size(0), -1)
        out = self.linear(out.view(out.size(0), -1))
        if out_feature == False:
            return out
        else:
            return out, feature


def ResNet18_8x(num_bits: int, num_classes: int = 10):
    return ResNet(num_bits, BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34_8x(num_bits: int, num_classes: int = 10):
    return ResNet(num_bits, BasicBlock, [3, 4, 6, 3], num_classes)


def ResNet50_8x(num_bits: int, num_classes: int = 10):
    return ResNet(num_bits, Bottleneck, [3, 4, 6, 3], num_classes)


def ResNet101_8x(num_bits: int, num_classes: int = 10):
    return ResNet(num_bits, Bottleneck, [3, 4, 23, 3], num_classes)


def ResNet152_8x(num_bits: int, num_classes: int = 10):
    return ResNet(num_bits, Bottleneck, [3, 8, 36, 3], num_classes)
