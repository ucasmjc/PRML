from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *
import torch.nn as nn
import torch.nn.functional as F


@MODELS.register_module()
class LightHead(BaseDecodeHead):
    """
    SEA-Former: Squeeze-enhanced Axial Transformer for Mobile Semantic Segmentation
    """

    def __init__(self, **kwargs):
        super(LightHead, self).__init__( **kwargs)

        self.conv1=ConvModule(self.in_channels,self.in_channels,3,1,1,bias=False,norm_cfg=self.norm_cfg)
        self.proj=nn.Conv2d(self.in_channels,self.num_classes,1,1,0)

    def forward(self, inputs):
        xx = self._transform_inputs(inputs)
        x_detail = xx[0]
        for i in range(len(self.embed_dims)):
            fuse = getattr(self, f"fuse{i + 1}")
            x_detail = fuse(x_detail, xx[i + 1])
        _c = self.linear_fuse(x_detail)
        x = self.cls_seg(_c)
        return x