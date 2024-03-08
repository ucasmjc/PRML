from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmengine.runner import CheckpointLoader
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import OptConfigType
from ..utils import DAPPM, PAPPM, BasicBlock, Bottleneck


@MODELS.register_module()
class PP_humanseg(BaseModule):
    def __init__(self,
                 in_channels,
                 align_corners=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        self.stem = ConvModule(in_channels, 36, 3, 2, 1, bias=False, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.max = nn.MaxPool2d(3, stride=2, padding=1)
        self.bn1 = ConvModule(36, 18, 1, 1, 0, bias=False, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.stage1 = nn.Sequential(shuffle_block(36, 2, norm_cfg, act_cfg, init_cfg),
                                    shuffle_block(72, 1, norm_cfg, act_cfg, init_cfg),
                                    shuffle_block(72, 1, norm_cfg, act_cfg, init_cfg),
                                    shuffle_block(72, 1, norm_cfg, act_cfg, init_cfg), )
        self.stage2 = nn.Sequential(shuffle_block(72, 2, norm_cfg, act_cfg, init_cfg),
                                    shuffle_block(144, 1, norm_cfg, act_cfg, init_cfg),
                                    shuffle_block(144, 1, norm_cfg, act_cfg, init_cfg),
                                    shuffle_block(144, 1, norm_cfg, act_cfg, init_cfg),
                                    shuffle_block(144, 1, norm_cfg, act_cfg, init_cfg),
                                    shuffle_block(144, 1, norm_cfg, act_cfg, init_cfg),
                                    shuffle_block(144, 1, norm_cfg, act_cfg, init_cfg),
                                    shuffle_block(144, 1, norm_cfg, act_cfg, init_cfg), )
        self.dw1 = nn.Sequential(nn.Conv2d(144, 144, 3, 1, 1, 1, 144), nn.BatchNorm2d(144),
                                 ConvModule(144, 64, 1, 1, 0, bias=False, norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.dw2 = nn.Sequential(nn.Conv2d(82, 82, 3, 1, 1, 1, 82), nn.BatchNorm2d(82),
                                 ConvModule(82, 64, 1, 1, 0, bias=False, norm_cfg=norm_cfg, act_cfg=act_cfg))

    def forward(self, x):
        x = self.stem(x)
        shortcut = self.bn1(x)
        x=self.max(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.dw1(x)
        x = F.interpolate(x, shortcut.shape[2:], mode='bilinear', align_corners=self.align_corners)
        x = torch.cat([shortcut, x], dim=1)
        out = self.dw2(x)
        return out


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class shuffle_block(BaseModule):
    def __init__(self,
                 in_channels: int,
                 stride,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.stride = stride
        if stride == 1:
            in_channels = in_channels // 2
            self.in_channel = int(in_channels)
        else:
            self.in_channel = in_channels
        self.dw = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, stride, 1, 1, in_channels),
                                nn.BatchNorm2d(in_channels),
                                ConvModule(in_channels, self.in_channel, 1, 1, 0, bias=False, norm_cfg=norm_cfg,
                                           act_cfg=act_cfg))
        self.conv1 = ConvModule(in_channels, self.in_channel, 1, 1, 0, bias=False, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.dw1 = nn.Sequential(nn.Conv2d(self.in_channel, self.in_channel, 3, stride, 1, 1, self.in_channel),
                                 nn.BatchNorm2d(self.in_channel),
                                 ConvModule(self.in_channel, self.in_channel, 1, 1, 0, bias=False, norm_cfg=norm_cfg,
                                            act_cfg=act_cfg))

    def forward(self, x):
        if self.stride == 1:
            shortcut, branch = x.chunk(2, dim=1)
        else:
            branch = x
            shortcut = self.dw(x)
        branch = self.dw1(self.conv1(branch))
        out = torch.cat([shortcut, branch], dim=1)
        return channel_shuffle(out, 2)
