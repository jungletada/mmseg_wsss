import math
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, to_2tuple
from mmseg.registry import MODELS

from .mcta_modules import DownConv, SemanticAttnModule
from .mcta_modules import SpatialPriorGNN
from .mct_vit import MCTViT


@MODELS.register_module()
class MCTAdapter(MCTViT):
    """Multi-scale Graph Attention Vision Transformer."""
    def __init__(self, *args, decay_parameter=0.996, input_size=512, **kwargs):
        """
        Args:
            *args: Variable length argument list.
            decay_parameter (float): Decay parameter for the GWRP.
            input_size (int): Size of the input image.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.stages = 4
        interval = int(self.depth // self.stages)
        self.stage_indices = tuple(i for i in range(0, self.depth + 1, interval))
        self.input_size = input_size
        img_size = to_2tuple(input_size)
        patch_size = to_2tuple(self.patch_embed.patch_size)
        self.Hp = math.ceil(img_size[0] / patch_size[0])
        self.Wp = math.ceil(img_size[1] / patch_size[1])
        self.num_patches = self.Hp * self.Wp
        self.spatial_dims = [self.embed_dim] * self.stages

        self.dilations = [1, 2, 3, 4]
        self.num_knn = [18, 15, 12, 9]
        self.spatial_scales = [16, 16, 32, 64]

        self.spatial_strides = [
            self.spatial_scales[i+1] // self.spatial_scales[i]
            for i in range(len(self.spatial_scales)-1)]
        
        spt_strides=[self.spatial_scales[0]//4] + self.spatial_strides
        self.spatial_prior = SpatialPriorGNN(
            inplanes=96,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            spt_strides=spt_strides)
                
        self.decay_parameter = decay_parameter
        self.spatial_sizes = [(math.ceil(img_size[0] / scale), math.ceil(img_size[1] / scale)) 
                              for scale in self.spatial_scales]
        self.sptial_pos_embed = [nn.Parameter(
            torch.zeros(1, self.spatial_dims[i], self.spatial_sizes[i][0], self.spatial_sizes[i][1]))
                for i in range(self.stages)]
        
        for i in range(self.stages):
            trunc_normal_(self.sptial_pos_embed[i], std=.02)
            
        self.cls_token = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.pos_embed_cls = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.pos_embed_pat = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed_cls, std=.02)
        trunc_normal_(self.pos_embed_pat, std=.02)
        
        self.proj_cls_embed = nn.Linear(
            self.stages, self.num_classes)

        self.spatial_fuse = nn.ModuleList([
            SemanticAttnModule(
                query_dim=self.spatial_dims[i],
                key_dim=self.embed_dim,
                num_classes=self.num_classes,
                num_heads=self.num_heads,
                attn_drop=0.,
                proj_drop=0.,
                drop_path=0.,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for i in range(self.stages)])

        self.down_convs = nn.ModuleList([
            DownConv(
                in_dim=self.spatial_dims[i],
                out_dim=self.spatial_dims[i+1],
                stride=self.spatial_strides[i])
            for i in range(self.stages - 1)])

        # self.channel_reduction = nn.Sequential(
        #     nn.Conv2d(self.embed_dim * 5, self.embed_dim, 1),
        #     nn.BatchNorm2d(self.embed_dim),
        #     nn.GELU())
        
        self.weights = nn.ParameterList([
            nn.Parameter(torch.zeros(1, self.num_classes, 1))
            for _ in range(self.stages)])
        
    def interpolate_pos_encoding(self, patch_tokens, token_size):
        """
        Interpolate position encoding for backbone tokens
        """
        if self.Hp == token_size[0] and self.Wp == token_size[1]:
            return self.pos_embed_pat
        patch_pos_embed = self.pos_embed_pat
        
        dim = patch_tokens.shape[-1]
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, self.Hp, self.Wp, dim).permute(0, 3, 1, 2),
            size=token_size,
            mode='bilinear',
            align_corners=False)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def interpolate_spatial_pos_encoding(self, x_spatial):
        """
        Interpolate position encoding for spatial tokens
        """
        # out_sizes = [((H+1)//scale, (W+1)//scale) for scale in self.spatial_scales]
        spatial_pos_embed = []
        for i in range(self.stages):
            spatial_pos_embed.append(
                F.interpolate(
                    self.sptial_pos_embed[i],
                    size=x_spatial[i].shape[2:],
                    mode='bilinear',
                    align_corners=False))
        return spatial_pos_embed
    
    def gwr_pooling_top_k(self, x, K=5):
        """
        Input:
            x -> B x K x Hp x Wp
            K -> Number of top values to weight as 1
        Return:
            out -> B x K
        """
        B, C, Hp, Wp = x.shape
        N = Hp * Wp
        flatten_x = x.view(B, C, -1).permute(0, 2, 1)  # B x (Hp x Wp) x K
        sorted_x, _ = torch.sort(flatten_x, dim=-2, descending=True)
        # Create weights for Top-K values (set to 1)
        top_k_weights = torch.ones(K, device=x.device)  # Weight of 1 for Top-K values
        # Create weights for the remaining values
        remaining_weights = torch.logspace(
            start=K, end=N-1, steps=N-K, base=self.decay_parameter, device=x.device)
        # Combine the Top-K and remaining weights
        all_weights = torch.cat((top_k_weights, remaining_weights), dim=0)
        # Broadcast weights to match the dimensions of sorted_x
        weights = all_weights.unsqueeze(0).unsqueeze(-1)  # Shape: 1 x N x 1
        # Apply the weights to sorted_x and compute the weighted sum
        out = torch.sum(sorted_x * weights, dim=-2) / weights.sum()

        return out
    
    def build_class_tokens(self, x):
        """
        Input: x -> list[B x C x H^ x W^]
        Return: cls-tokens -> B x Cls x C
        """
        x_cls = [self.gwr_pooling_top_k(f).unsqueeze(-1) for f in x]  # list [B x C x 1]
        x_cls = torch.cat(x_cls, dim=-1)        # B x C x 4
        x_cls = self.proj_cls_embed(x_cls)      # B x C x Cls
        x_cls = x_cls.permute(0, 2, 1).contiguous()  # B x Cls x C
        return x_cls
     
    def forward_features(self, x):
        """
        Input:
            x: B x 3 x H x W
        Return:
            x_cls: [B x K x C] 
            x_vit: [B x Np x C]
            attn_weights: list[B x Hd x N' x N']
            x_spatial: list[B x C x H^ x W^]
        """
        b, _, H, W = x.shape                # B x 3 x H x W
        x_spatial = self.spatial_prior(x)   # list [B x C x H^ x W^]
        x = self.patch_embed(x)
        token_size = (H // self.patch_embed.patch_size[0], W // self.patch_embed.patch_size[1])

        if not self.training:
            pos_embed_pat = self.interpolate_pos_encoding(x, token_size=token_size)
            x = x + pos_embed_pat
            sptial_pos_embed = self.interpolate_spatial_pos_encoding(x_spatial)
            for i in range(self.stages):
                x_spatial[i] += sptial_pos_embed[i].to(x.device)
        else:
            x = x + self.pos_embed_pat
            for i in range(self.stages):
                x_spatial[i] += self.sptial_pos_embed[i].to(x.device)

        nn_cls_tokens = self.cls_token.expand(b, -1, -1) + self.pos_embed_cls
        cls_tokens = self.build_class_tokens(x_spatial) + nn_cls_tokens

        x = torch.cat((cls_tokens, x), dim=1) # Concat input with Nc class tokens
        x = self.pos_drop(x)                  # B x (N') x C, where N' = Nc + Np

        attn_weights = []
        #-------------------  Modify block for ablation study -------------------#
        for i in range(self.stages):
            for j in range(self.stage_indices[i], self.stage_indices[i+1]):
                x, weights_j = self.blocks[j](x)
                attn_weights.append(weights_j)

            cls_stru, x_spatial[i] = self.spatial_fuse[i](
                x_spatial=x_spatial[i],
                x_backbone=x,
                token_size=token_size)
            # zero initialized weights for adding new class tokens
            x_cls = x[:, :self.num_classes] + self.weights[i] * cls_stru
            x_vit = x[:, self.num_classes:]
            x = torch.cat((x_cls, x_vit), dim=1)

            if i != self.stages - 1:
                z = self.down_convs[i](x_spatial[i])
                x_spatial[i + 1] = x_spatial[i + 1] + z

        return {
            'x_cls': x[:, :self.num_classes], 
            'x_vit': x[:, self.num_classes:], 
            'attn': attn_weights, 
            'x_branch': x_spatial
            }
    
    def reshape_patch_tokens(self, patch_tokens, H, W):
        """
        Reshape patch tokens from [B, Np, C] to [B, C, Hp, Wp]
        """
        B, _, C = patch_tokens.shape
        Hp = H // self.patch_embed.patch_size[0]
        Wp = W // self.patch_embed.patch_size[1]
        patch_tokens = torch.reshape(patch_tokens, [B, Hp, Wp, C])
        patch_tokens = patch_tokens.permute([0, 3, 1, 2]).contiguous() # B x C x Hp x Wp
        return patch_tokens

    def forward(self, x):
        """
        Basic forward for training image classification.
        """
        b, _, h, w = x.shape
        # basic forward
        feat_dict = self.forward_features(x)
        # class tokens
        last_cls_tokens = feat_dict['x_cls'] # [B, K, C]
        cls_logits = last_cls_tokens.mean(-1) # [B, K]
        
        x_vit = self.reshape_patch_tokens(
            feat_dict['x_vit'], h, w) # [B, C, Hp, Wp]
        x_out = [x_vit]
        out_size = x_vit.shape[2:]
        for feat in feat_dict['x_branch']:
            feat = F.interpolate(
                feat,
                size=out_size,
                mode="bilinear",
                align_corners=False)
            x_out.append(feat)

        return tuple(x_out)
