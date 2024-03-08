from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead
import torch.nn as nn
import torch.nn.functional as F


@MODELS.register_module()
class ppHead(BaseDecodeHead):

    def __init__(self,in_channels: int,
                 channels: int,
                 num_classes: int,align_corners=None, **kwargs):
        super(ppHead, self).__init__(in_channels,
            channels,
            num_classes=num_classes,**kwargs)
        self.align_corners = align_corners
        self.conv = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, 1, 64), nn.BatchNorm2d(64),
                                  ConvModule(64, num_classes, 1, 1, 0, bias=False,
                                             norm_cfg=dict(type='BN', requires_grad=True)))

    def forward(self, inputs):
        x = self.conv(inputs)
        H, W = x.shape[2:]
        x = F.interpolate(x, (H * 2, W * 2), mode='bilinear', align_corners=self.align_corners)
        return x
