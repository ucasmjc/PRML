from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead
from ..utils import PAPPM, DAPPM,cusPPM
from ..backbones.official import Block
from mmseg.models.utils import *
import torch.nn as nn
import torch.nn.functional as F
from .psp_head import PPM_changed
import torch
from mmseg.models.losses import accuracy
from .util import AlignedModule,AlignedModulev2,AlignedModulev2PoolingAtten,PagFM,paperPagFM,seafusion,seashortcut,sPagFM,s1PagFM,custom
@MODELS.register_module()
class testHead(BaseDecodeHead):

    def __init__(self, neck,fuse, **kwargs):
        super(testHead, self).__init__(**kwargs)
        necklist = [PAPPM(48, 24, 48, 5), DAPPM(48, 24, 48, 5),
                    PPM_changed((1, 2, 3, 6), 48, [8,4,2,2], conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=dict(type='ReLU', inplace=True),
                        align_corners=self.align_corners),cusPPM(48,24,48,5)]  # params:1.715-0.979,
        fuselist = [PagFM(48,16,32),seafusion(48,16,32),paperPagFM(48,16,12),seashortcut(48,16,32),paperPagFM(48,16,32),sPagFM(48,16,32),AlignedModule(48,16,32),AlignedModulev2(48,16,32),AlignedModulev2PoolingAtten(48,16,32),s1PagFM(48,16,32),
        custom(48,16,32)]
        self.fuse4 =fuselist[fuse]
        self.neck = necklist[neck]

    def forward(self, inputs):
        xx = self._transform_inputs(inputs)
        xx32 = self.neck(xx[1])
        out=self.fuse4(xx[0],xx32)
        x = self.cls_seg(out)
        return x

@MODELS.register_module()
class bdHead(BaseDecodeHead):
    """
    SEA-Former: Squeeze-enhanced Axial Transformer for Mobile Semantic Segmentation
    """

    def __init__(self, neck,fuse, **kwargs):
        super(bdHead, self).__init__(**kwargs)
        necklist = [PAPPM(48, 24, 48, 5), DAPPM(48, 24, 48, 5),
                    PPM_changed((1, 2, 3, 6), 48, [8,4,2,2], conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=dict(type='ReLU', inplace=True),
                        align_corners=self.align_corners),cusPPM(48,24,48,5)]  # params:1.715-0.979,
        fuselist = [PagFM(48,16,32),seafusion(48,16,32),paperPagFM(48,16,12),seashortcut(48,16,32),paperPagFM(48,16,32),sPagFM(48,16,32),AlignedModule(48,16,32),AlignedModulev2(48,16,32),AlignedModulev2PoolingAtten(48,16,32),s1PagFM(48,16,32),
        custom(48,16,32)]
        self.fuse4 =fuselist[fuse]
        self.neck = necklist[neck]
        self.conv=nn.Sequential(ConvModule(16,32,3,1,1),ConvModule(32,32,3,1,1),nn.Conv2d(32,1,1,1,0))

    def forward(self, inputs):
        xx = self._transform_inputs(inputs)
        xx32 = self.neck(xx[1])
        out=self.fuse4(xx[0],xx32)
        x = self.cls_seg(out)
        if self.training:
            bd=self.conv(xx[0])
            return x, bd
        else:
            return x
        
    def _stack_batch_gt(self, batch_data_samples):
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        gt_edge_segs = [
            data_sample.gt_edge_map.data for data_sample in batch_data_samples
        ]
        gt_sem_segs = torch.stack(gt_semantic_segs, dim=0)
        gt_edge_segs = torch.stack(gt_edge_segs, dim=0)
        return gt_sem_segs, gt_edge_segs

    def loss_by_feat(self, seg_logits,
                     batch_data_samples) -> dict:
        loss = dict()
        semlogit, bdlogit = seg_logits
        sem_label, bd_label = self._stack_batch_gt(batch_data_samples)
        semlogit = resize(
            input=semlogit,
            size=sem_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        bdlogit = resize(
            input=bdlogit,
            size=bd_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        sem_label = sem_label.squeeze(1)
        bd_label = bd_label.squeeze(1)
        loss['loss_sem'] = self.loss_decode[0](
            semlogit, sem_label, ignore_index=self.ignore_index)
        loss['loss_bd'] = self.loss_decode[1](bdlogit, bd_label)
        loss['acc_seg'] = accuracy(
            semlogit, sem_label, ignore_index=self.ignore_index)
        return loss