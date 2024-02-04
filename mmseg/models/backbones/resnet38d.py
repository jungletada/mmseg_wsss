import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmengine.model import BaseModule
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmseg.registry import MODELS
import warnings


class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1, 
                 first_dilation=None, dilation=1, norm_cfg=dict(type='BN'),):
        super(ResBlock, self).__init__()

        self.same_shape = (in_channels == out_channels and stride == 1)

        if first_dilation == None: first_dilation = dilation

        _, self.bn_branch2a = build_norm_layer(norm_cfg, in_channels)

        self.conv_branch2a = nn.Conv2d(in_channels, mid_channels, 3, stride,
                                       padding=first_dilation, dilation=first_dilation, bias=False)

        _, self.bn_branch2b1 = build_norm_layer(norm_cfg, mid_channels)

        self.conv_branch2b1 = nn.Conv2d(mid_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False)

        if not self.same_shape:
            self.conv_branch1 = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)

    def forward(self, x, get_x_bn_relu=False):

        branch2 = self.bn_branch2a(x)
        branch2 = F.relu(branch2)

        x_bn_relu = branch2

        if not self.same_shape:
            branch1 = self.conv_branch1(branch2)
        else:
            branch1 = x

        branch2 = self.conv_branch2a(branch2)
        branch2 = self.bn_branch2b1(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.conv_branch2b1(branch2)

        x = branch1 + branch2

        if get_x_bn_relu:
            return x, x_bn_relu

        return x

    def __call__(self, x, get_x_bn_relu=False):
        return self.forward(x, get_x_bn_relu=get_x_bn_relu)


class ResBlock_bot(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, dropout=0., norm_cfg=dict(type='BN'),):
        super(ResBlock_bot, self).__init__()
        self.same_shape = (in_channels == out_channels and stride == 1)
        _, self.bn_branch2a = build_norm_layer(norm_cfg, in_channels)
        self.conv_branch2a = nn.Conv2d(in_channels, out_channels//4, 1, stride, bias=False)

        _, self.bn_branch2b1 = build_norm_layer(norm_cfg, out_channels//4)
        self.dropout_2b1 = torch.nn.Dropout2d(dropout)
        self.conv_branch2b1 = nn.Conv2d(out_channels//4, out_channels//2, 3, padding=dilation, dilation=dilation, bias=False)

        _, self.bn_branch2b2 = build_norm_layer(norm_cfg, out_channels//2)
        self.dropout_2b2 = torch.nn.Dropout2d(dropout)
        self.conv_branch2b2 = nn.Conv2d(out_channels//2, out_channels, 1, bias=False)

        if not self.same_shape:
            self.conv_branch1 = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)

    def forward(self, x, get_x_bn_relu=False):
        branch2 = self.bn_branch2a(x)
        branch2 = F.relu(branch2)
        x_bn_relu = branch2

        branch1 = self.conv_branch1(branch2)

        branch2 = self.conv_branch2a(branch2)

        branch2 = self.bn_branch2b1(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.dropout_2b1(branch2)
        branch2 = self.conv_branch2b1(branch2)

        branch2 = self.bn_branch2b2(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.dropout_2b2(branch2)
        branch2 = self.conv_branch2b2(branch2)

        x = branch1 + branch2

        if get_x_bn_relu:
            return x, x_bn_relu

        return x

    def __call__(self, x, get_x_bn_relu=False):
        return self.forward(x, get_x_bn_relu=get_x_bn_relu)


@MODELS.register_module()
class ResNet38(BaseModule):
    def __init__(self,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 pretrained=None,
                 init_cfg=None):
        super(ResNet38, self).__init__(init_cfg)
        
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        
        self.norm_cfg = norm_cfg
            
        self.conv1a = nn.Conv2d(3, 64, 3, padding=1, bias=False)

        self.b2 = ResBlock(64, 128, 128, stride=2, norm_cfg=self.norm_cfg)
        self.b2_1 = ResBlock(128, 128, 128, norm_cfg=self.norm_cfg)
        self.b2_2 = ResBlock(128, 128, 128, norm_cfg=self.norm_cfg)

        self.b3 = ResBlock(128, 256, 256, stride=2, norm_cfg=self.norm_cfg)
        self.b3_1 = ResBlock(256, 256, 256, norm_cfg=self.norm_cfg)
        self.b3_2 = ResBlock(256, 256, 256, norm_cfg=self.norm_cfg)

        self.b4 = ResBlock(256, 512, 512, stride=2, norm_cfg=self.norm_cfg)
        self.b4_1 = ResBlock(512, 512, 512, norm_cfg=self.norm_cfg)
        self.b4_2 = ResBlock(512, 512, 512, norm_cfg=self.norm_cfg)
        self.b4_3 = ResBlock(512, 512, 512, norm_cfg=self.norm_cfg)
        self.b4_4 = ResBlock(512, 512, 512, norm_cfg=self.norm_cfg)
        self.b4_5 = ResBlock(512, 512, 512, norm_cfg=self.norm_cfg)

        self.b5 = ResBlock(512, 512, 1024, stride=1, first_dilation=1, dilation=2, norm_cfg=self.norm_cfg)
        self.b5_1 = ResBlock(1024, 512, 1024, dilation=2, norm_cfg=self.norm_cfg)
        self.b5_2 = ResBlock(1024, 512, 1024, dilation=2, norm_cfg=self.norm_cfg)

        self.b6 = ResBlock_bot(1024, 2048, stride=1, dilation=4, dropout=0.3, norm_cfg=self.norm_cfg)
        self.b7 = ResBlock_bot(2048, 4096, dilation=4, dropout=0.5, norm_cfg=self.norm_cfg)
        _, self.bn7 = build_norm_layer(self.norm_cfg, 4096)
        self.not_training = [self.conv1a]
    
    def forward(self, x):
        x = self.conv1a(x)
        
        x = self.b2(x)
        x = self.b2_1(x)
        x = self.b2_2(x)

        x = self.b3(x)
        x = self.b3_1(x)
        x = self.b3_2(x)

        x = self.b4(x)
        x = self.b4_1(x)
        x = self.b4_2(x)
        x = self.b4_3(x)
        x = self.b4_4(x)
        x = self.b4_5(x)

        x, conv4 = self.b5(x, get_x_bn_relu=True)
        x = self.b5_1(x)
        x = self.b5_2(x)

        x, conv5 = self.b6(x, get_x_bn_relu=True)

        x = self.b7(x)
        conv6 = F.relu(self.bn7(x))
        
        return (conv4, conv5, conv6)
        #return dict({'conv4': conv4, 'conv5': conv5, 'conv6': conv6})

    def train(self, mode=True):
        
        super().train(mode)
        
        for layer in self.not_training:
            if isinstance(layer, nn.Conv2d):
                layer.weight.requires_grad = False
            elif isinstance(layer, nn.Module):
                for c in layer.children():
                    c.weight.requires_grad = False
                    if c.bias is not None:
                        c.bias.requires_grad = False

        # for layer in self.modules():
        #     if isinstance(layer, nn.BatchNorm2d):
        #         layer.eval()
        #         layer.bias.requires_grad = False
        #         layer.weight.requires_grad = False
