import math
import warnings

import torch
import torch.nn as nn
from adaptive_instance_normalization import AdaptiveInstanceNormalization
from destylization_module import DeStylizationModule
from normalization import AdaIN
from torch.nn import Linear
from sequential import DestylerSequential


class IBN(nn.Module):
    r"""Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`
    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """

    def __init__(self, planes, ratio=0.5):
        super(IBN, self).__init__()
        self.half = int(planes * ratio)
        # self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.IN = AdaptiveInstanceNormalization(planes - self.half)
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x, y):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous(), y)
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class BasicBlock_IBN(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, ibn=None, stride=1, downsample=None):
        super(BasicBlock_IBN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        if ibn == 'a':
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.IN = nn.InstanceNorm2d(
            planes, affine=True) if ibn == 'b' else None
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.IN is not None:
            out = self.IN(out)
        out = self.relu(out)

        return out


class Bottleneck_IBN(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ibn=None, stride=1, downsample=None):
        super(Bottleneck_IBN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if ibn == 'a':
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.IN = nn.InstanceNorm2d(
            planes * 4, affine=True) if ibn == 'b' else None
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, y = None):
        residual = x

        out = self.conv1(x)
        if y is None:
            out = self.bn1(out)
        else:
            out = self.bn1(out, y)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.IN is not None:
            out = self.IN(out)
        out = self.relu(out)

        return out

class ResNet_IBN(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 ibn_cfg=('a', 'a', 'a', None),
                 num_classes=1000):
        self.inplanes = 64
        super(ResNet_IBN, self).__init__()

        self.ds = DeStylizationModule()
        if ibn_cfg[0] == 'a':
            self.ds_fc_5_1 = Linear(32, 64)
            self.ds_fc_5_2 = Linear(32, 64)
            self.ds_fc_5_3 = Linear(32, 64)
            self.ds_fc_5_4 = Linear(32, 128)
            self.ds_fc_5_5 = Linear(32, 128)
            self.ds_fc_5_6 = Linear(32, 128)
            self.ds_fc_5_7 = Linear(32, 128)
            self.ds_fc_5_8 = Linear(32, 256)
            self.ds_fc_5_9 = Linear(32, 256)
            self.ds_fc_5_10 = Linear(32, 256)
            self.ds_fc_5_11 = Linear(32, 256)
            self.ds_fc_5_12 = Linear(32, 256)
            self.ds_fc_5_13 = Linear(32, 256)


        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        if ibn_cfg[0] == 'b':
            self.bn1 = nn.InstanceNorm2d(64, affine=True)
        else:
            self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], ibn=ibn_cfg[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, ibn=ibn_cfg[1])
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, ibn=ibn_cfg[2])
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, ibn=ibn_cfg[3])
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, ibn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes,
                            None if ibn == 'b' else ibn,
                            stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                None if (ibn == 'b' and i < blocks-1) else ibn))

        return DestylerSequential(*layers)

    def forward(self, x):

        feat = self.ds(x)
        y_1 = self.ds_fc_5_1(feat)
        y_2 = self.ds_fc_5_2(feat)
        y_3 = self.ds_fc_5_3(feat)
        y_4 = self.ds_fc_5_4(feat)
        y_5 = self.ds_fc_5_5(feat)
        y_6 = self.ds_fc_5_6(feat)
        y_7 = self.ds_fc_5_7(feat)
        y_8 = self.ds_fc_5_8(feat)
        y_9 = self.ds_fc_5_9(feat)
        y_10 = self.ds_fc_5_10(feat)
        y_11 = self.ds_fc_5_11(feat)
        y_12 = self.ds_fc_5_12(feat)
        y_13 = self.ds_fc_5_13(feat)


        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x, [y_1, y_2, y_3])
        x = self.layer2(x, [y_4, y_5, y_6, y_7])
        x = self.layer3(x, [y_8, y_9, y_10, y_11, y_12, y_13])
        x = self.layer4(x, [])

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
