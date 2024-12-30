# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import torch
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin, Conv2d


##############################
#    Basic layers
##############################
def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def norm_layer(norm, nc):
    # normalization layer 2d
    norm = norm.lower()
    if norm == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return layer


class MLP(Seq):
    def __init__(self, channels, act='relu', norm=None, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(Lin(channels[i - 1], channels[i], bias))
            if act is not None and act.lower() != 'none':
                m.append(act_layer(act))
            if norm is not None and norm.lower() != 'none':
                m.append(norm_layer(norm, channels[-1]))
        super(MLP, self).__init__(*m)


class BasicConv(Seq):
    """Basic Conv Module
    (Conv + Norm + Act) x N
    """
    def __init__(self, channels, act='relu', norm=None, groups=6, bias=True, drop=0.):
        m = []
        for i in range(1, len(channels)):
            # print(f"In Basic Conv, {channels[i - 1]} and {channels[i]}; heads={groups}")
            m.append(Conv2d(channels[i - 1], channels[i], 1, bias=bias, groups=groups)) # groups=4
            if norm is not None and norm.lower() != 'none':
                m.append(norm_layer(norm, channels[-1]))
            if act is not None and act.lower() != 'none':
                m.append(act_layer(act))
            if drop > 0:
                m.append(nn.Dropout2d(drop))

        super(BasicConv, self).__init__(*m)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                if hasattr(m, 'weight') and m.weight is not None:
                    m.weight.data.fill_(1)
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.data.zero_()
                

def batched_index_select(x, idx):
    r"""fetches neighbors features from a given neighbor idx
    Args:
        x (Tensor): input feature Tensor
                x -> B x C x N x 1
        idx (Tensor): edge index
                idx -> B x N x k
    Returns:
        Tensor: output neighbors features
            feature -> B x C x N x k
    """
    batch_size, num_dims, num_vertices_reduced = x.shape[:3]
    _, num_vertices, k = idx.shape
    idx_batch = torch.arange(0, batch_size, device=idx.device)
    idx_batch = idx_batch.view(-1, 1, 1) * num_vertices_reduced # B x 1 x 1
    idx = idx + idx_batch   # {B x N x k} + {B x 1 x 1}
    idx = idx.contiguous().view(-1)     # {B x N x k}
    
    x = x.transpose(2, 1).contiguous()  # B x N x C x 1
    feature = x.view(batch_size * num_vertices_reduced, -1)[idx, :] # (B x N) x C
    feature = feature.view(batch_size, num_vertices, k, num_dims).permute(0, 3, 1, 2).contiguous()
    
    return feature
