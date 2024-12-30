import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from mmengine.model import BaseModule

from .gcn_lib import Grapher


def nlc2nchw(x, d_size):
    _, N, C = x.shape
    assert d_size[0] * d_size[1] == N, f"{d_size} not equal to {N}"
    x = x.permute(0, 2, 1).reshape(-1, C, d_size[0], d_size[1]).contiguous()
    return x


def nchw2nlc(x):
    B, C = x.shape[:2]
    x = x.permute(0, 2, 3, 1).reshape(B, -1, C).contiguous()
    return x  
    

class DownConv(BaseModule):
    """
    Downsampling Convolutional Layer with optional normalization and activation.
    This layer applies a convolution operation followed by normalization and activation.
    """
    def __init__(self, in_dim, out_dim=None, kernel_size=3, stride=2, padding=1,
                 norm_layer=nn.BatchNorm2d):
        super(DownConv, self).__init__()
        out_dim = in_dim if out_dim is None else out_dim
        self.conv = nn.Conv2d(
            in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = norm_layer(out_dim)
        self.act = nn.GELU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x


class SpatialPriorGNN(BaseModule):
    def __init__(self,
                 inplanes=96,
                 embed_dim=384,
                 num_heads=6,
                 knn=[18, 15, 12, 9],
                 dilation=[4, 3, 2, 1],
                 spt_strides=[4, 2, 2, 1],
                 norm_layer=nn.BatchNorm2d,
                 act_layer=nn.GELU):
        super().__init__()
        conv_type = 'mr'
        assert inplanes % num_heads == 0, f"{inplanes} should be divided by {num_heads}"
        self.stem = nn.Sequential(*[ # downsample by 4
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(inplanes),
            act_layer(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)])
        
        if spt_strides[0] == 4:
            self.conv2 = nn.Sequential(*[
                nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(2 * inplanes),
                nn.Conv2d(2 * inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(2 * inplanes),
                act_layer(),
            ])
        elif spt_strides[0] == 2:
            self.conv2 = nn.Sequential(*[
                nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(2 * inplanes),
                act_layer(),
            ])
        else: raise NotImplementedError

        self.gnn_2 = Grapher(
            in_channels=2 * inplanes, 
            kernel_size=knn[0], 
            dilation=dilation[0], 
            conv=conv_type,
            groups=num_heads)
        
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=spt_strides[1], padding=1, bias=False),
            norm_layer(4 * inplanes),
            act_layer()])

        self.gnn_3 = Grapher(
            in_channels=4 * inplanes, 
            kernel_size=knn[1], 
            dilation=dilation[1], 
            conv=conv_type, 
            groups=num_heads)
        
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=spt_strides[2], padding=1, bias=False),
            norm_layer(4 * inplanes),
            act_layer(),
        ])

        self.gnn_4 = Grapher(
            in_channels=4 * inplanes, 
            kernel_size=knn[2], 
            dilation=dilation[2], 
            conv=conv_type, 
            groups=num_heads)
        
        self.conv5 = nn.Sequential(*[
            nn.Conv2d(4 * inplanes, 8 * inplanes, kernel_size=3, stride=spt_strides[3], padding=1, bias=False),
            norm_layer(8 * inplanes),
            act_layer(),
        ])
        
        self.gnn_5 = Grapher(
            in_channels=8 * inplanes, 
            kernel_size=knn[3], 
            dilation=dilation[3], 
            conv=conv_type,
            groups=num_heads)
        
        # self.fc1 = nn.Conv2d(inplanes, embed_dims[0], kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc5 = nn.Conv2d(8 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        c1 = self.stem(x)

        c2 = self.conv2(c1)
        c2 = self.gnn_2(c2)

        c3 = self.conv3(c2)
        c3 = self.gnn_3(c3)

        c4 = self.conv4(c3)
        c4 = self.gnn_4(c4)

        c5 = self.conv5(c4)
        c5 = self.gnn_5(c5)

        # c1 = self.fc1(c1) # 4s
        c2 = self.fc2(c2) # 8s
        c3 = self.fc3(c3) # 16s
        c4 = self.fc4(c4) # 32s
        c5 = self.fc5(c5) # 64s
    
        return [c2, c3, c4, c5]


class CrossAttention(BaseModule):
    """
    Cross-Attention for spatial-backbone tokening mixing
    """
    def __init__(self, query_dim, key_dim, num_classes=20, num_heads=8, qkv_bias=False, 
                 qk_scale=None, attn_drop=0., proj_drop=0., mask_ratio=0.3):
        super().__init__()
        self.num_heads = num_heads
        dim = min(query_dim, key_dim)
        self.head_dim = dim // num_heads
        self.mask_ratio = mask_ratio
        self.scale = qk_scale or self.head_dim ** -0.5
        
        self.proj_q = nn.Linear(query_dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(key_dim, dim * 2, bias=qkv_bias)
        self.proj_cls = nn.Linear(key_dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, query_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.Cls = num_classes

    def forward(self, input_query, input_key):
        B, Nt, _ = input_query.shape
        N = input_key.shape[1]
        # Concat class tokens to query
        cls_tokens = self.proj_cls(input_key[:, :self.Cls, :])
        input_query = self.proj_q(input_query)
        input_query = torch.cat((cls_tokens, input_query), dim=1)
        q = input_query.reshape(
            B, (self.Cls+Nt), self.num_heads, self.head_dim).permute(0, 2, 1, 3)
       
        kv = self.proj_kv(input_key).reshape(
            B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1] # B x Nd x (Cls+N) x d
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B x Nd x (Cls+Nt) x (Cls+N)
        #== Split class and patch tokens ============================================#               
        attn_cls, attn_pat = torch.split(attn, [self.Cls, N-self.Cls], dim=-1)
        attn_pat = attn_pat.softmax(dim=-1)
        attn_cls = attn_cls.softmax(dim=-1)
        attn = torch.cat((attn_cls, attn_pat), dim=-1)
        #======================================================================#     
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape( # B x Nd x (Cls+Nt) x d
            B, (self.Cls+Nt), -1)               # B x (Cls+Nt) x Ct
        x = self.proj(x)
        x = self.proj_drop(x)
        
        x = x[:, self.Cls:, :]
        return x


class MLP(BaseModule):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SemanticAttnModule(BaseModule):
    """
    Semantic Cross-Attention for Feature fusion
    """
    def __init__(
            self,
            query_dim,
            key_dim,
            num_classes=20,
            num_heads=6,
            attn_drop=0.,
            proj_drop=0.,
            drop_path=0.,
            mask_ratio=0.,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=nn.LayerNorm,):
        super().__init__()
        self.norm1 = norm_layer(query_dim)
        self.norm2 = norm_layer(key_dim)
        self.norm3 = norm_layer(query_dim)

        self.num_heads = num_heads
        self.dim = min(query_dim, key_dim)
        self.head_dim = self.dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.nc = num_classes
        self.mask_ratio = mask_ratio

        self.proj_q = nn.Linear(query_dim, self.dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(key_dim, self.dim * 2, bias=qkv_bias)
        self.proj_cls = nn.Linear(key_dim, self.dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim, query_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.mlp = MLP(
            in_features=self.dim,
            hidden_features=self.dim * 4,
            out_features=self.dim)

    def forward_attention_gnn(self, input_query, input_key, token_size, spatial_size):
        """
        Cross-Attention for spatial token (query) and patch token (key)
        """
        B, Ni, _ = input_query.shape
        N = input_key.shape[1] - self.nc
        cls_tokens = self.proj_cls(input_key[:, :self.nc, :]) # Concat class tokens to query
        input_query = self.proj_q(input_query)
        input_query = torch.cat((cls_tokens, input_query), dim=1)
        q = input_query.reshape(
            B, (self.nc + Ni), self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.proj_kv(input_key).reshape(
            B, (self.nc + N), 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1] # B x Nd x (Cls+N) x d
        # k[:, :, :self.nc, :] = q[:, :, :self.nc, :] # Attn_{qq}
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B x Nd x (Cls+Ni) x (Cls+N)
        #===============================================================#
        attn_cls, attn_pat = torch.split(attn, [self.nc, N], dim=-1)
        attn_pat = attn_pat.softmax(dim=-1)
        attn_cls = attn_cls.softmax(dim=-1)
        attn = torch.cat((attn_cls, attn_pat), dim=-1)
        # attn = attn.softmax(dim=-1) # traditional softmax
        #===============================================================#
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape( # B x Nd x (Cls+Nt) x d
            B, (self.nc+Ni), -1)               # B x (Cls+Ni) x Ct
        x = self.proj(x)
        x = self.proj_drop(x)

        return x[:, :self.nc], x[:, self.nc:]

    def forward(self, x_spatial, x_backbone, token_size):
        """
        Input:
            x_spatial: spatial features from prior module->[B, Cq, Hi, Wi]
            x_backbone: multi-class token transformer features->[B, (Cls+N'), Ck]
        Output:
            x_spatial: updated spatial features 
        """
        h, w = x_spatial.shape[2:]
        x_spatial = nchw2nlc(x_spatial)
        x_cls = x_backbone[:, :self.nc]
        x_cat = torch.cat((x_cls, x_spatial), dim=1).clone() # B, (Cls+Ni), C

        x_cls, x_spatial = self.forward_attention_gnn(
            self.norm1(x_spatial),
            self.norm2(x_backbone),
            token_size=token_size,
            spatial_size=(h, w))

        x_cat = x_cat + self.drop_path(torch.cat((x_cls, x_spatial), dim=1))
        x_cat = x_cat + self.drop_path(self.mlp(self.norm3(x_cat)))

        x_cls, x_spatial = x_cat[:, :self.nc], x_cat[:, self.nc:]
        x_spatial = nlc2nchw(x_spatial, d_size=(h, w))

        return x_cls, x_spatial
    