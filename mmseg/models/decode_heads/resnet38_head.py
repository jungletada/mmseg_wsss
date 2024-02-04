import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead


@MODELS.register_module()
class SimpleHead(BaseDecodeHead):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fc8_seg_conv1 = nn.Conv2d(4096, 512, (3, 3), stride=1, padding=12, dilation=12, bias=True)
        torch.nn.init.xavier_uniform_(self.fc8_seg_conv1.weight)

        self.fc8_seg_conv2 = nn.Conv2d(512, self.num_classes, (3, 3), stride=1, padding=12, dilation=12, bias=True)
        torch.nn.init.xavier_uniform_(self.fc8_seg_conv2.weight)

    def forward(self, inputs):
        x = inputs[-1]
        x_seg = F.relu(self.fc8_seg_conv1(x))
        x_seg = self.fc8_seg_conv2(x_seg)
        return x_seg

