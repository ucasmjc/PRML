import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model.base_module import BaseModule
from torch.utils import checkpoint as cp
from ..builder import BACKBONES
from mmcv.cnn import build_norm_layer




@BACKBONES.register_module()
class SeaFormer(BaseModule):
    def __init__(self,
                 norm_cfg,
                 drop=0,
                 drop_ratio=0,
                 init_cfg=None):
        super(SeaFormer, self).__init__(init_cfg=init_cfg)
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        if self.init_cfg is not None:
            self.pretrained = self.init_cfg['checkpoint']
        # stem layer
        self.stem = self._make_stem_layer(5)
        self.stage1 = nn.Sequential(MBv2(32, 64, 3, 2, 3), MBv2(64, 64, 3, 1, 3))
        self.stage2 = nn.Sequential(MBv2(64, 128, 5, 2, 3), SeaBlock(128, 16, 256, 2, norm_cfg, drop_path=drop_ratio, drop=drop),
                                    SeaBlock(128, 16, 256, 2, norm_cfg, drop_path=drop_ratio, drop=drop))
        self.stage3 = nn.Sequential(MBv2(128, 160, 3, 2, 6), SeaBlock(160, 24, 640, 2, norm_cfg, drop_path=drop_ratio, drop=drop),
                                    SeaBlock(160, 24, 640, 2, norm_cfg, drop_path=drop_ratio, drop=drop))
        self.f2 = Fusion_Block(128, 32, 64, norm_cfg)
        self.f3 = Fusion_Block(160, 64, 96, norm_cfg)

    def forward(self, x):
        x = self.stem(x)
        xh1 = self.stage1(x)
        xh2 = self.stage2(xh1)
        xl1 = self.f2(xh2, x)
        xh3 = self.stage3(xh2)
        out = self.f3(xh3, xl1)
        return out

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            checkpoint = _load_checkpoint(self.pretrained, logger=logger, map_location='cpu')
            if 'state_dict_ema' in checkpoint:
                state_dict = checkpoint['state_dict_ema']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            self.load_state_dict(state_dict, False)

    def _make_stem_layer(self, num_blocks: int) -> nn.Sequential:
        """Make stem layer.
        Args:
            num_blocks (int): Number of blocks.
        Returns:
            nn.Sequential: The stem layer.
        """

        layers = [ConvModule(
            3,
            16,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg)]
        in_channel = [16, 16, 16, 16, 32]
        out_channel = [16, 16, 16, 32, 32]
        kernal_Size = [3, 3, 3, 5, 5]
        stride = [1, 2, 1, 2, 1]
        expansion = [1, 4, 3, 3, 3]
        for i in range(num_blocks):
            layers.append(MBv2(in_channel[i], out_channel[i], kernal_Size[i], stride[i], expansion[i],
                               norm_cfg=self.norm_cfg, ))

        return nn.Sequential(*layers)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class SeaAttention(BaseModule):
    def __init__(self,
                 in_channels: int,
                 key_channels: int,
                 norm_cfg,
                 head=4,
                 init_cfg=None):
        super().__init__(init_cfg)
        dh=key_channels * head
        self.head = head
        self.dim = key_channels * head
        self.k = nn.Sequential(nn.Conv2d(in_channels,dh,1,bias=False),
                               build_norm_layer(norm_cfg, dh)[1])
        self.q = nn.Sequential(nn.Conv2d(in_channels, dh, 1,bias=False),
                               build_norm_layer(norm_cfg, dh)[1])
        self.v = nn.Sequential(nn.Conv2d(in_channels, key_channels * head * 2, 1,bias=False),
                               build_norm_layer(norm_cfg, key_channels * head * 2)[1])
        self.projw = ConvModule(key_channels * head * 2, key_channels * head * 2, 1, norm_cfg=norm_cfg,
                                act_cfg=dict(type="ReLU6"),
                                order=('act', 'conv', 'norm'))
        self.projh = ConvModule(key_channels * head * 2, key_channels * head * 2, 1, norm_cfg=norm_cfg,
                                act_cfg=dict(type="ReLU6"),
                                order=('act', 'conv', 'norm'))
        self.scale = key_channels ** -0.5
        self.dwconv = ConvModule(key_channels * head * 4, key_channels * head * 4, 3, padding=1,
                                 groups=key_channels * head * 4,
                                 norm_cfg=norm_cfg, act_cfg=dict(type="ReLU6"))
        self.proj1 = nn.Conv2d(key_channels * head * 4, in_channels, 1,bias=False)
        self.bn = build_norm_layer(norm_cfg, in_channels)[1]
        self.proj2 = ConvModule(key_channels * head * 2, in_channels, 1, norm_cfg=None, act_cfg=None)
        self.kw = nn.Parameter(torch.randn([1, key_channels * head, 16], requires_grad=True))
        self.qw = nn.Parameter(torch.randn([1, key_channels * head, 16], requires_grad=True))
        self.kh = nn.Parameter(torch.randn([1, key_channels * head, 16], requires_grad=True))
        self.qh = nn.Parameter(torch.randn([1, key_channels * head, 16], requires_grad=True))

    def forward(self, x):
        B, C, H, W = x.shape
        k = self.k(x)
        q = self.q(x)
        v = self.v(x)
        kw = (torch.mean(k, dim=2) + F.interpolate(self.kw, (W), mode='linear', align_corners=False)).reshape(B,
                                                                                                              self.head,
                                                                                                              -1, W)
        qw = (torch.mean(q, dim=2) + F.interpolate(self.qw, (W), mode='linear', align_corners=False)).reshape(B,
                                                                                                              self.head,
                                                                                                              -1,
                                                                                                              W).permute(
            0,
            1,
            3,
            2)
        vw = torch.mean(v, dim=2).reshape(B, self.head, -1, W).permute(0, 1, 3, 2)
        attw = torch.softmax(torch.matmul(qw, kw) * self.scale, dim=-1)
        outw = torch.matmul(attw, vw).permute(0, 2, 1, 3).reshape(B, 1, W, -1)
        outw = self.projw(outw.permute(0, 3, 1, 2))
        # or outw=torch.matmul(attw,vw).permute(0,1,3,2).reshape(B,-1,1,W)
        kh = (torch.mean(k, dim=3) + F.interpolate(self.kh, (H), mode='linear', align_corners=False)).reshape(B,
                                                                                                              self.head,
                                                                                                              -1, H)
        qh = (torch.mean(q, dim=3) + F.interpolate(self.qh, (H), mode='linear', align_corners=False)).reshape(B,
                                                                                                              self.head,
                                                                                                              -1,
                                                                                                              H).permute(
            0,
            1,
            3,
            2)
        vh = torch.mean(v, dim=3).reshape(B, self.head, -1, H).permute(0, 1, 3, 2)
        atth = torch.softmax(torch.matmul(qh, kh) * self.scale, dim=-1)
        outh = torch.matmul(atth, vh).permute(0, 2, 1, 3).reshape(B, 1, H, -1)
        outh = self.projh(outh.permute(0, 3, 2, 1))
        out = outw.add(outh)
        # detail
        sigmoid = h_sigmoid()
        xx = torch.cat([k, q, v], dim=1)
        detail = self.bn(self.proj1(self.dwconv(xx)))
        out = detail * sigmoid(self.proj2(out + v))
        return x + out


class SeaBlock(BaseModule):
    def __init__(self,
                 in_channels: int,
                 key_channels: int,
                 emb_channels: int,
                 num_block,
                 norm_cfg,
                 drop=0,
                 drop_path=0,
                 head=4,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.att = SeaAttention(in_channels=in_channels, key_channels=key_channels, norm_cfg=norm_cfg, head=head)
        self.num = num_block
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop = nn.Dropout(drop)
        self.FFN = nn.Sequential(nn.Conv2d(in_channels, emb_channels, 1,bias=False), build_norm_layer(norm_cfg, emb_channels)[1],
                                 nn.Conv2d(emb_channels, emb_channels, 3, padding=1, groups=emb_channels,bias=False),
                                 nn.ReLU6(), self.drop, nn.Conv2d(emb_channels, in_channels, 1,bias=False),
                                 build_norm_layer(norm_cfg, in_channels)[1], self.drop)

    def forward(self, x):
        x = self.drop_path(self.att(x)) + x
        x = self.drop_path(self.FFN(x)) + x
        return x


class Fusion_Block(BaseModule):
    def __init__(self,
                 h_channels: int,
                 l_channels: int,
                 emb_channel: int,
                 norm_cfg,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.emb_h = ConvModule(h_channels, emb_channel, 1, norm_cfg=norm_cfg, act_cfg=None)
        self.emb_l = ConvModule(l_channels, emb_channel, 1, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x_h, x_l):
        sigmoid = h_sigmoid()
        fh = self.emb_h(x_h)
        fl = self.emb_l(x_l)
        sigma = F.interpolate(sigmoid(fh), size=fl.shape[2:],
                              mode="bilinear",
                              align_corners=False)
        out = sigma * fl
        return out


class MBv2(nn.Module):
    """InvertedResidual block for MobileNetV2.

    Args:
        in_channels (int): The input channels of the InvertedResidual block.
        out_channels (int): The output channels of the InvertedResidual block.
        stride (int): Stride of the middle (first) 3x3 convolution.
        expand_ratio (int): Adjusts number of channels of the hidden layer
            in InvertedResidual by this amount.
        dilation (int): Dilation rate of depthwise conv. Default: 1
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU6').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.

    Returns:
        Tensor: The output tensor.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernal_size,
                 stride,
                 expand_ratio,
                 dilation=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 with_cp=False,
                 **kwargs):
        super(MBv2, self).__init__()
        self.stride = stride
        assert stride in [1, 2], f'stride must in [1, 2]. ' \
                                 f'But received {stride}.'
        self.with_cp = with_cp
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))

        layers = []
        if expand_ratio != 1:
            layers.append(
                ConvModule(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    **kwargs))
        layers.extend([
            ConvModule(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=kernal_size,
                stride=stride,
                padding=kernal_size // 2,
                dilation=dilation,
                groups=hidden_dim,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                **kwargs),
            ConvModule(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None,
                **kwargs)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):

        def _inner_forward(x):
            if self.use_res_connect:
                return x + self.conv(x)
            else:
                return self.conv(x)

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6
