import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mmengine.model import BaseModule


class Mlp(BaseModule):
    def __init__(self, in_features, hidden_features=None, 
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
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


class Attention(BaseModule):
    """
    Base Attention for DeiT.
    """
    def __init__(self, dim, num_heads=6, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., num_classes=20):
        super().__init__()
        self.nc = num_classes
        self.n_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape  # Here N = #patches + #class-tokens
        qkv = self.qkv(x).reshape(
            B, N, 3, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # d for each head, Nd heads in total. --> B x Nd x N x d for {q, k, v}.
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B x Nd x N x N
        #======================================================================#
        attn_cls, attn_pat = torch.split(attn, [self.nc, N-self.nc], dim=-1)
        attn_pat = attn_pat.softmax(dim=-1)
        attn_cls = attn_cls.softmax(dim=-1)
        attn = torch.cat((attn_cls, attn_pat), dim=-1)
        # attn = attn.softmax(dim=-1) # change to vanilla Softmax
        #======================================================================#
        weights = attn
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, weights


class Block(BaseModule):
    """
    Transformer Block requires both patch tokens and class tokens
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_classes=20):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            num_classes=num_classes)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        o, weights = self.attn(self.norm1(x))
        x = x + self.drop_path(o)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, weights


class PatchEmbed(BaseModule):
    """ Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, output_type='1d'):
        x = self.proj(x) # B, C, H, W
        if output_type == '2d':
            return x
        return x.flatten(2).transpose(1, 2)


class MCTViT(BaseModule):
    """
    DeiT with Multi-class Tokens
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=384, depth=12,
                 num_heads=6, mlp_ratio=4, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), mask_type=None,):
        super().__init__()
        self.num_classes = num_classes
        self.mask_type = mask_type
        self.img_size = img_size
        self.patch_size = patch_size
        self.depth = depth
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.num_heads = num_heads
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(
            0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        blocks = []
        for i in range(depth):
            blocks.append(
                Block(dim=embed_dim,num_heads=num_heads,mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                        qk_scale=qk_scale,drop=drop_rate,attn_drop=attn_drop_rate,drop_path=dpr[i],
                        norm_layer=norm_layer,num_classes=num_classes))
        self.blocks = nn.ModuleList(blocks)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}
