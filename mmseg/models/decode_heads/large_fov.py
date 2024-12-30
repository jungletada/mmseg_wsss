import torch
import torch.nn as nn
from mmseg.registry import MODELS
from mmcv.cnn import build_norm_layer
from .decode_head import BaseDecodeHead
from ..utils import resize


def conv3x3(in_planes, out_planes, stride=1, dilation=1, padding=1):
    " 3 x 3 conv"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=False
    )


def conv1x1(in_planes, out_planes, stride=1, dilation=1, padding=1):
    " 1 x 1 conv"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, dilation=dilation, bias=False)


@MODELS.register_module()
class LargeFOV(BaseDecodeHead):
    def __init__(self, in_channels, dilation=5, **kwargs):
        super(LargeFOV, self).__init__(
            in_channels, **kwargs)
        self.dilation = dilation
        
        self.channel_reduction = nn.Sequential(
            nn.Conv2d(in_channels * 5, in_channels, 1),
            build_norm_layer(self.norm_cfg, in_channels)[1],
            nn.GELU())

        self.conv6 = conv3x3(
            in_planes=in_channels, 
            out_planes=self.channels, 
            padding=self.dilation, 
            dilation=self.dilation)
        self.act6 = nn.GELU()

        self.conv7 = conv3x3(
            in_planes=self.channels, 
            out_planes=self.channels, 
            padding=self.dilation, 
            dilation=self.dilation)
        self.act7 = nn.GELU()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
        return None

    def _forward_feature(self, inputs):
        x = torch.cat(inputs, dim=1)
        x = resize(
            x,
            scale_factor=4,
            mode='bilinear',
            align_corners=self.align_corners
        )
        x = self.channel_reduction(x)
        x = self.conv6(x)
        x = self.act6(x)
        x = self.conv7(x)
        x = self.act7(x)
        return x
        
    def forward(self, inputs):
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output