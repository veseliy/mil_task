import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import numpy as np


class BaseBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(BaseBlock, self).__init__()
        self.subsampling = 2 if in_channels != out_channels else 1
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.subsampling, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):

        out = self.sequential(x)
        return out

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, base_block = BaseBlock):

        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.base_block = base_block(in_channels, out_channels)
        self.block_expantion = self.base_block.subsampling
        self.shortcut = nn.Identity()
        if self.base_block.subsampling==2:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride=self.block_expantion,
                          padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )



    def forward(self, x):

        residual = self.shortcut(x)
        out = self.base_block(x)
        out += residual
        out = F.relu(out)

        return out

class ResLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResBlock, n=1):
        super(ResLayer, self).__init__()
        self.layer = nn.Sequential(
            block(in_channels, out_channels),
                *[block(out_channels, out_channels) for _ in range(n-1)]
            )
    def forward(self, x):
        out = self.layer(x)
        return out

class ResEnter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResEnter, self).__init__()
        self.enter_seq = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),

        )
    def forward(self, x):
        out = self.enter_seq(x)
        return out

class ResExit(nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features, n_classes, bias=False)
    def forward(self, x):
        out = self.avg(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class ResMidLine(nn.Module):
    def __init__(self, sizes=[16, 32, 64], depths = [2, 2, 2], block=ResBlock):
        super().__init__()
        self.in_out_block_sizes = list(zip(sizes, sizes[1:]))
        self.blocks = nn.ModuleList([
            ResLayer(sizes[0], sizes[0], n=depths[0], block=block),
            *[ResLayer(in_channels, out_channels, n=n, block=block)
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, depths[1:])]
        ])
    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x

class ResNet(nn.Module):
    def __init__(self, sizes=[16, 32, 64], depths = [2, 2, 2]):
        super(ResNet, self).__init__()
        self.enter = ResEnter(3, sizes[0])
        self.mid = ResMidLine(sizes=sizes, depths = depths)
        self.exit = ResExit(64, 10)
    def forward(self, x):
        out = self.enter(x)
        out = self.mid(out)
        out = self.exit(out)
        return out
