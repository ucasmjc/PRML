import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from .light import h_sigmoid
from ..utils import SELayer
class paperPagFM(BaseModule):
    """Pixel-attention-guided fusion module.

    Args:
        in_channels (int): The number of input channels.
        channels (int): The number of channels.
        after_relu (bool): Whether to use ReLU before attention.
            Default: False.
        with_channel (bool): Whether to use channel attention.
            Default: False.
        upsample_mode (str): The mode of upsample. Default: 'bilinear'.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(typ='ReLU', inplace=True).
        init_cfg (dict): Config dict for initialization. Default: None.
    """

    def __init__(self,
                 high: int,
                 low,
                 channels: int,
                 after_relu: bool = False,
                 with_channel: bool = False,
                 upsample_mode: str = 'bilinear',
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(typ='ReLU', inplace=True),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.after_relu = after_relu
        self.with_channel = with_channel
        self.upsample_mode = upsample_mode
        self.conv1= ConvModule(
            high, low, 1, norm_cfg=norm_cfg, act_cfg=None)
        self.f_i = ConvModule(
            low, channels, 1, norm_cfg=norm_cfg, act_cfg=None)
        self.f_p = ConvModule(
            low, channels, 1, norm_cfg=norm_cfg, act_cfg=None)
        if with_channel:
            self.up = ConvModule(
                channels, channels, 1, norm_cfg=norm_cfg, act_cfg=None)
        if after_relu:
            self.relu = MODELS.build(act_cfg)

    def forward(self, x_p, x_i) :
        """Forward function.

        Args:
            x_p (Tensor): The featrue map from P branch.
            x_i (Tensor): The featrue map from I branch.

        Returns:
            Tensor: The feature map with pixel-attention-guided fusion.
        """
        if self.after_relu:
            x_p = self.relu(x_p)
            x_i = self.relu(x_i)
        x_i=self.conv1(x_i)
        f_i = self.f_i(x_i)
        f_i = F.interpolate(
            f_i,
            size=x_p.shape[2:],
            mode=self.upsample_mode,
            align_corners=False)

        f_p = self.f_p(x_p)

        if self.with_channel:
            sigma = torch.sigmoid(self.up(f_p * f_i))
        else:
            sigma = torch.sigmoid(torch.sum(f_p * f_i, dim=1).unsqueeze(1))
        x_i = F.interpolate(
            x_i,
            size=x_p.shape[2:],
            mode=self.upsample_mode,
            align_corners=False)
        out = sigma * x_i + (1 - sigma) * x_p
        return out

class PagFM(BaseModule):
    """Pixel-attention-guided fusion module.

    Args:
        in_channels (int): The number of input channels.
        channels (int): The number of channels.
        after_relu (bool): Whether to use ReLU before attention.
            Default: False.
        with_channel (bool): Whether to use channel attention.
            Default: False.
        upsample_mode (str): The mode of upsample. Default: 'bilinear'.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(typ='ReLU', inplace=True).
        init_cfg (dict): Config dict for initialization. Default: None.
    """

    def __init__(self,
                 high: int,
                 low,
                 channels,
                 after_relu: bool = False,
                 with_channel: bool = False,
                 upsample_mode: str = 'bilinear',
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(typ='ReLU', inplace=True),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.after_relu = after_relu
        self.with_channel = with_channel
        self.upsample_mode = upsample_mode
        self.f_i = ConvModule(
            high, channels, 1, norm_cfg=norm_cfg, act_cfg=None)
        self.f_p = ConvModule(
            low, channels, 1, norm_cfg=norm_cfg, act_cfg=None)
        if with_channel:
            self.up = ConvModule(
                channels, channels, 1, norm_cfg=norm_cfg, act_cfg=None)
        if after_relu:
            self.relu = MODELS.build(act_cfg)

    def forward(self, x_p, x_i) :
        """Forward function.

        Args:
            x_p (Tensor): The featrue map from P branch.
            x_i (Tensor): The featrue map from I branch.

        Returns:
            Tensor: The feature map with pixel-attention-guided fusion.
        """
        if self.after_relu:
            x_p = self.relu(x_p)
            x_i = self.relu(x_i)

        f_i = self.f_i(x_i)
        f_i = F.interpolate(
            f_i,
            size=x_p.shape[2:],
            mode=self.upsample_mode,
            align_corners=False)

        f_p = self.f_p(x_p)
        if self.with_channel:
            sigma = torch.sigmoid(self.up(f_p * f_i))
        else:
            sigma = torch.sigmoid(torch.sum(f_p * f_i, dim=1).unsqueeze(1))

        out = sigma * f_i + (1 - sigma) * f_p
        return out

class seafusion(nn.Module):
    def __init__(
            self,
            high: int,
            low: int,
            embed_dim: int,
            norm_cfg=dict(type='BN', requires_grad=True),
            activations=None,
    ) -> None:
        super(seafusion, self).__init__()
        self.norm_cfg = norm_cfg
        self.local_embedding = ConvModule(low, embed_dim, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_act = ConvModule(high, embed_dim, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.act = h_sigmoid()

    def forward(self, x_l, x_g):
        '''
        x_g: global features
        x_l: local features
        '''
        B, C, H, W = x_l.shape
        B, C_c, H_c, W_c = x_g.shape
        local_feat = self.local_embedding(x_l)
        global_act = self.global_act(x_g)
        sig_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)
        out = local_feat * sig_act
        return out
class seashortcut(nn.Module):
    def __init__(
            self,
            high: int,
            low: int,
            embed_dim: int,
            norm_cfg=dict(type='BN', requires_grad=True),
            activations=None,
    ) -> None:
        super(seashortcut, self).__init__()
        self.norm_cfg = norm_cfg
        self.local_embedding = ConvModule(low, embed_dim, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_act = ConvModule(high, embed_dim, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.act = h_sigmoid()
        self.conv=ConvModule(high, embed_dim, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)

    def forward(self, x_l, x_g):
        '''
        x_g: global features
        x_l: local features
        '''
        B, C, H, W = x_l.shape
        B, C_c, H_c, W_c = x_g.shape
        local_feat = self.local_embedding(x_l)
        global_act = self.global_act(x_g)
        sig_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)
        x_g=F.interpolate(x_g, size=(H, W), mode='bilinear', align_corners=False)
        out = local_feat * sig_act
        return out+self.conv(x_g)
class sPagFM(BaseModule):


    def __init__(self,
                 high: int,
                 low,
                 channels,
                 after_relu: bool = False,
                 with_channel: bool = False,
                 upsample_mode: str = 'bilinear',
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(typ='ReLU', inplace=True),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.after_relu = after_relu
        self.with_channel = with_channel
        self.upsample_mode = upsample_mode
        self.conv=ConvModule(
            high, low, 1, norm_cfg=norm_cfg, act_cfg=None)
        self.f_i = ConvModule(
            high, channels, 1, norm_cfg=norm_cfg, act_cfg=None)
        self.f_p = ConvModule(
            low, channels, 1, norm_cfg=norm_cfg, act_cfg=None)
        if with_channel:
            self.up = ConvModule(
                channels, channels, 1, norm_cfg=norm_cfg, act_cfg=None)
        if after_relu:
            self.relu = MODELS.build(act_cfg)

    def forward(self, x_p, x_i) :
        """Forward function.

        Args:
            x_p (Tensor): The featrue map from P branch.
            x_i (Tensor): The featrue map from I branch.

        Returns:
            Tensor: The feature map with pixel-attention-guided fusion.
        """
        if self.after_relu:
            x_p = self.relu(x_p)
            x_i = self.relu(x_i)

        f_i = self.f_i(x_i)
        f_i = F.interpolate(
            f_i,
            size=x_p.shape[2:],
            mode=self.upsample_mode,
            align_corners=False)

        f_p = self.f_p(x_p)
        if self.with_channel:
            sigma = torch.sigmoid(self.up(f_p * f_i))
        else:
            sigma = torch.sigmoid(torch.sum(f_p * f_i, dim=1).unsqueeze(1))
        x_i= F.interpolate(
            x_i,
            size=x_p.shape[2:],
            mode=self.upsample_mode,
            align_corners=False)
        f_i=self.conv(x_i)
        out = sigma * f_i + (1 - sigma) * x_p
        return out
class s1PagFM(BaseModule):
    """Pixel-attention-guided fusion module.

    Args:
        in_channels (int): The number of input channels.
        channels (int): The number of channels.
        after_relu (bool): Whether to use ReLU before attention.
            Default: False.
        with_channel (bool): Whether to use channel attention.
            Default: False.
        upsample_mode (str): The mode of upsample. Default: 'bilinear'.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(typ='ReLU', inplace=True).
        init_cfg (dict): Config dict for initialization. Default: None.
    """

    def __init__(self,
                 high: int,
                 low,
                 channels,
                 after_relu: bool = False,
                 with_channel: bool = False,
                 upsample_mode: str = 'bilinear',
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(typ='ReLU', inplace=True),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.after_relu = after_relu
        self.with_channel = with_channel
        self.upsample_mode = upsample_mode
        self.conv=ConvModule(
            low, high, 1, norm_cfg=norm_cfg, act_cfg=None)
        self.f_i = ConvModule(
            high, channels, 1, norm_cfg=norm_cfg, act_cfg=None)
        self.f_p = ConvModule(
            low, channels, 1, norm_cfg=norm_cfg, act_cfg=None)
        if with_channel:
            self.up = ConvModule(
                channels, channels, 1, norm_cfg=norm_cfg, act_cfg=None)
        if after_relu:
            self.relu = MODELS.build(act_cfg)

    def forward(self, x_p, x_i) :
        """Forward function.

        Args:
            x_p (Tensor): The featrue map from P branch.
            x_i (Tensor): The featrue map from I branch.

        Returns:
            Tensor: The feature map with pixel-attention-guided fusion.
        """
        if self.after_relu:
            x_p = self.relu(x_p)
            x_i = self.relu(x_i)

        f_i = self.f_i(x_i)
        f_i = F.interpolate(
            f_i,
            size=x_p.shape[2:],
            mode=self.upsample_mode,
            align_corners=False)

        f_p = self.f_p(x_p)
        if self.with_channel:
            sigma = torch.sigmoid(self.up(f_p * f_i))
        else:
            sigma = torch.sigmoid(torch.sum(f_p * f_i, dim=1).unsqueeze(1))
        x_i= F.interpolate(
            x_i,
            size=x_p.shape[2:],
            mode=self.upsample_mode,
            align_corners=False)
        f_p=self.conv(x_p)
        out = sigma * x_i + (1 - sigma) * f_p
        return out

class AlignedModule(nn.Module):

    def __init__(self, high,low, embed, kernel_size=3):
        super(AlignedModule, self).__init__()
        self.neck = nn.Conv2d(low, high, 1, bias=False)
        self.down_h = nn.Conv2d(high, embed, 1, bias=False)
        self.down_l = nn.Conv2d(high, embed, 1, bias=False)
        self.flow_make = nn.Conv2d(embed*2, 2, kernel_size=kernel_size, padding=1, bias=False)

    def forward(self, low_feature,h_feature):
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature=self.neck(low_feature)
        l_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature = F.interpolate(h_feature, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([h_feature, l_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)

        return h_feature+low_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output


class AlignedModulev2(nn.Module):

    def __init__(self, high,low, embed, kernel_size=3):
        super(AlignedModulev2, self).__init__()
        self.neck=nn.Conv2d(low,high,1,bias=False)
        self.down_h = nn.Conv2d(high, embed, 1, bias=False)
        self.down_l = nn.Conv2d(high, embed, 1, bias=False)
        self.flow_make = nn.Conv2d(embed*2, 4, kernel_size=kernel_size, padding=1, bias=False)
        self.flow_gate = nn.Sequential(
            nn.Conv2d(embed * 2, 1, kernel_size=kernel_size, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self,low_feature,h_feature):
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature=self.neck(low_feature)
        l_feature = self.down_l(low_feature)
        h_feature= self.down_h(h_feature)
        h_feature = F.interpolate(h_feature, size=size, mode="bilinear", align_corners=True)

        flow = self.flow_make(torch.cat([h_feature, l_feature], 1))
        flow_up, flow_down = flow[:, :2, :, :] , flow[:, 2:, :, :]

        h_feature_warp = self.flow_warp(h_feature_orign, flow_up, size=size)
        l_feature_warp = self.flow_warp(low_feature, flow_down, size=size)

        flow_gates = self.flow_gate(torch.cat([h_feature, l_feature], 1))

        fuse_feature = h_feature_warp * flow_gates + l_feature_warp * (1 - flow_gates)

        return fuse_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output


class AlignedModulev2PoolingAtten(nn.Module):

    def __init__(self, high, low, embed, kernel_size=3):
        super(AlignedModulev2PoolingAtten, self).__init__()
        self.neck = nn.Conv2d(low, high, 1, bias=False)
        self.down_h = nn.Conv2d(high, embed, 1, bias=False)
        self.down_l = nn.Conv2d(high, embed, 1, bias=False)
        self.flow_make = nn.Conv2d(embed * 2, 4, kernel_size=kernel_size, padding=1, bias=False)
        self.flow_gate = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=kernel_size, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self,low_feature,h_feature):
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature=self.neck(low_feature)
        l_feature = self.down_l(low_feature)
        h_feature= self.down_h(h_feature)
        h_feature = F.interpolate(h_feature, size=size, mode="bilinear", align_corners=True)

        flow = self.flow_make(torch.cat([h_feature, l_feature], 1))
        flow_up, flow_down = flow[:, :2, :, :] , flow[:, 2:, :, :]

        h_feature_warp = self.flow_warp(h_feature_orign, flow_up, size=size)
        l_feature_warp = self.flow_warp(low_feature, flow_down, size=size)

        h_feature_mean = torch.mean(h_feature, dim=1).unsqueeze(1)
        l_feature_mean = torch.mean(low_feature, dim=1).unsqueeze(1)
        h_feature_max = torch.max(h_feature, dim=1)[0].unsqueeze(1)
        l_feature_max = torch.max(low_feature, dim=1)[0].unsqueeze(1)

        flow_gates = self.flow_gate(torch.cat([h_feature_mean, l_feature_mean, h_feature_max, l_feature_max], 1))

        fuse_feature = h_feature_warp * flow_gates + l_feature_warp * (1 - flow_gates)

        return fuse_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output
class custom(nn.Module):

    def __init__(self, high,low, embed, kernel_size=3):
        super(custom, self).__init__()
        self.neck=nn.Conv2d(low,high,1,bias=False)
        self.down_h = nn.Conv2d(high, embed, 1, bias=False)
        self.down_l = nn.Conv2d(high, embed, 1, bias=False)
        self.flow_make = nn.Conv2d(embed*2, 4, kernel_size=kernel_size, padding=1, bias=False)
        self.flow_gate = nn.Sequential(
            nn.Conv2d(embed * 2, 1, kernel_size=kernel_size, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.se=SELayer(high)

    def forward(self,low_feature,h_feature):
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature=self.neck(low_feature)
        l_feature = self.down_l(low_feature)
        h_feature= self.down_h(h_feature)
        h_feature = F.interpolate(h_feature, size=size, mode="bilinear", align_corners=True)

        flow = self.flow_make(torch.cat([h_feature, l_feature], 1))
        flow_up, flow_down = flow[:, :2, :, :] , flow[:, 2:, :, :]

        h_feature_warp = self.flow_warp(h_feature_orign, flow_up, size=size)
        l_feature_warp = self.flow_warp(low_feature, flow_down, size=size)

        flow_gates = self.flow_gate(torch.cat([h_feature, l_feature], 1))

        fuse_feature = h_feature_warp * flow_gates + l_feature_warp * (1 - flow_gates)
        out=self.se(fuse_feature)
        return out

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output
