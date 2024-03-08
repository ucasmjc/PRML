from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead


@MODELS.register_module()
class SeaHead(BaseDecodeHead):

    def __init__(self, **kwargs):
        super(SeaHead, self).__init__(**kwargs)
        self.conv = ConvModule(96, 96, 1, groups=96, norm_cfg=self.norm_cfg)

    def forward(self, inputs):
        x = self.conv(inputs)
        output = self.cls_seg(x)
        return output
