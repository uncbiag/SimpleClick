import torch.nn as nn

from .models_vit import Mlp


class CrossAttention(nn.Module):
    """ Multi-head cross-attention operation
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, target):
        assert query.shape == target.shape

        B, N, C = query.shape
        q = self.q(query).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k(target).reshape(B, N, self.num_heads, C // self.num_heads)
        v = self.v(target).reshape(B, N, self.num_heads, C // self.num_heads)

        q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1,2).contiguous().reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
    
        return out


class CrossBlock(nn.Module):
    """ Multi-head cross-attention block
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., mlp_drop=0., qkv_bias=False,
            attn_drop=0., proj_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                   attn_drop=attn_drop, proj_drop=proj_drop)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=mlp_drop)
        
    def forward(self, query, target):
        query = query + self.attn(self.norm1(query), target)
        query = query + self.mlp(self.norm2(query))
        return query