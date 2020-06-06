"""
    VGG model definition
    ported from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import math
import torch
import torch.nn as nn

import curves

INPUT_DIM = 32*32*3
N_HIDDEN_NODES = 5*INPUT_DIM
DROPOUT_PROB = .5

__all__ = ['OneLayer', 'OneLayerBN']


def make_layers(config, batch_norm=False, fix_points=None):
    layer_blocks = nn.ModuleList()
    activation_blocks = nn.ModuleList()
    poolings = nn.ModuleList()

    kwargs = dict()
    conv = nn.Conv2d
    bn = nn.BatchNorm2d
    if fix_points is not None:
        kwargs['fix_points'] = fix_points
        conv = curves.Conv2d
        bn = curves.BatchNorm2d

    in_channels = 3
    for sizes in config:
        layer_blocks.append(nn.ModuleList())
        activation_blocks.append(nn.ModuleList())
        for channels in sizes:
            layer_blocks[-1].append(conv(in_channels, channels, kernel_size=3, padding=1, **kwargs))
            if batch_norm:
                layer_blocks[-1].append(bn(channels, **kwargs))
            activation_blocks[-1].append(nn.ReLU(inplace=True))
            in_channels = channels
        poolings.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return layer_blocks, activation_blocks, poolings


class OneLayerBase(nn.Module):
    def __init__(self, num_classes, batch_norm=False):
        super(OneLayerBase, self).__init__()

        print("initialized OneLayer")

        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, N_HIDDEN_NODES),
            nn.ReLU(inplace=True),
            nn.Linear(N_HIDDEN_NODES, num_classes),
        )

    def forward(self, x):
        # print("-"*50)
        # print("data block size:")
        # print(x.size())
        # print("data:")
        # print(x)

        x = x.view(x.size(0), -1)
        x = self.net(x)

        # ones = torch.zeros([1, INPUT_DIM]).cuda()

        # print("dummy output:")
        # print(self.fc1(ones))
        # print("output:")
        # print(x)

        return x


class OneLayerCurve(nn.Module):
    def __init__(self, num_classes, fix_points, depth=16, batch_norm=False):
        super(OneLayerCurve, self).__init__()

        self.fc1 = curves.Linear(INPUT_DIM, N_HIDDEN_NODES, fix_points=fix_points)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = curves.Linear(N_HIDDEN_NODES, num_classes, fix_points=fix_points)

    def forward(self, x, coeffs_t):
        x = x.view(x.size(0), -1)

        x = self.fc1(x, coeffs_t)
        x = self.relu1(x)
        x = self.fc2(x, coeffs_t)

        return x


class OneLayer:
    base = OneLayerBase
    curve = OneLayerCurve
    kwargs = {
        'batch_norm': False
    }

class OneLayerBN:
    base = OneLayerBase
    curve = OneLayerCurve
    kwargs = {
        'batch_norm': True
    }
