import torch
import torch.nn as nn

from functools import partial
from .pos_embed import interpolate_pos_embed


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer() if act_layer else nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """ Multi-head self-attention operation
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1,2).contiguous().reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    """ Multi-head self-attention block 
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., mlp_drop=0., qkv_bias=False, 
            attn_drop=0., proj_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                              attn_drop=attn_drop, proj_drop=proj_drop)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=mlp_drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """ Patch embedding layer for 2D images
    """
    def __init__(self, img_size=(224,224), patch_size=(16,16), in_chans=3, 
                 embed_dim=768, norm_layer=None, flatten=False):
        super().__init__()
        self.in_chans = in_chans
        self.img_size = img_size
        self.patch_size = patch_size

        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, 
                              stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for global average pooling 
    """    
    def __init__(self, img_size=(224,224), patch_size=(16, 16), in_chans=3, 
                 embed_dim=768, depth=12, global_atten_freq=1, num_heads=12, 
                 mlp_ratio=4., qkv_bias=True, pos_drop_rate=0., attn_drop_rate=0.,
                 proj_drop_rate=0., norm_layer=None, act_layer=None):
        super().__init__()
        self.global_atten_freq = global_atten_freq
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, 
            in_chans=in_chans, embed_dim=embed_dim, flatten=True
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        norm_layer = norm_layer if norm_layer else partial(nn.LayerNorm, eps=1e-6)
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, attn_drop=attn_drop_rate, norm_layer=norm_layer, 
                proj_drop=proj_drop_rate, act_layer=act_layer
            ) for _ in range(depth)])

        self.fc_norm = norm_layer(embed_dim)

        self.init_weights()

    def init_weights_from_pretrained(self, pretrained_path):
        if pretrained_path:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            print("Load pre-trained checkpoint from: %s" % pretrained_path)
            checkpoint_model = checkpoint['model']

            # interpolate position embedding
            interpolate_pos_embed(self, checkpoint_model)

            # load pre-trained model
            msg = self.load_state_dict(checkpoint_model, strict=False)
            print(msg)

    def init_weights(self):
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively similar to normal_(std=0.02) 
        # as the default cutoff in trunc_normal_(std=.02) is too big (-2., 2.)
        nn.init.normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def no_weight_decay(self):
        return {'pos_embed', 'dist_token'}

    def window_split(self, x, x_size=(448, 448), win_size=(224, 224)):
        """ window splitting for window attention (the default window size is 224x224)
        input shape: (B_old, N, C)
        out shape: (B_new*win_h*win_w, N//(win_h*win_w), C)
        """
        B, N, C = x.shape
        grid_h = x_size[0] // self.patch_embed.patch_size[0]
        grid_w = x_size[1] // self.patch_embed.patch_size[1]
        assert N == grid_h * grid_w

        grid_h_per_window = win_size[0] // self.patch_embed.patch_size[0]
        grid_w_per_window = win_size[1] // self.patch_embed.patch_size[1]

        num_win_h = grid_h // grid_h_per_window
        num_win_w = grid_w // grid_w_per_window

        x = x.view(B, num_win_h, grid_h//num_win_h, num_win_w, grid_w//num_win_w, C)
        x_window_splitted = x.permute((0, 1, 3, 2, 4, 5)).contiguous()

        B_new = B * num_win_h * num_win_w
        N_new = grid_h * grid_w // (num_win_h * num_win_w)
        x_window_splitted = x_window_splitted.view(B_new, N_new, C)

        return x_window_splitted

    def window_split_reverse(self, x, x_size=(448, 448), win_size=(224, 224)):
        """ reverse the window splitting
        in shape: (B_old*win_h*win_w, N//(win_h*win_w), C)
        out shape: (B_new, N, C)
        """
        B, N, C = x.shape

        grid_h = x_size[0] // self.patch_embed.patch_size[0]
        grid_w = x_size[1] // self.patch_embed.patch_size[1]

        grid_h_per_window = win_size[0] // self.patch_embed.patch_size[0]
        grid_w_per_window = win_size[1] // self.patch_embed.patch_size[1]

        num_win_h = grid_h // grid_h_per_window
        num_win_w = grid_w // grid_w_per_window

        B_new = B // (num_win_h * num_win_w)
        N_new = grid_h * grid_w
        assert B_new * N_new == B * N

        x = x.view(B_new, num_win_h,num_win_w, grid_h//num_win_h, grid_w//num_win_w,C)
        x = x.permute((0, 1, 3, 2, 4, 5)).contiguous().view(B_new, N_new, C)

        return x

    def forward(self, x, other_feats=None, keep_shape=False):
        B, C, H, W = x.shape

        x = self.patch_embed(x)
        if other_feats is not None:
            x += other_feats

        x = self.pos_drop(x + self.pos_embed[:, 1:])

        num_blocks = len(self.blocks)
        if self.global_atten_freq <= 1:
            # perform global attention for all blocks
            for i in range(num_blocks):
                x = self.blocks[i](x)
        else:
            # perform global attention sparsely with window attention
            is_window_splitted = False
            for i in range(1, num_blocks + 1):
                is_global_block = i % self.global_atten_freq == 0 or i == num_blocks
                if is_global_block:
                    if is_window_splitted:
                        x = self.window_split_reverse(x, x_size=(H, W))
                        is_window_splitted = False
                else:
                    if not is_window_splitted:
                        x = self.window_split(x, x_size=(H,W))
                        is_window_splitted = True

                x = self.blocks[i-1](x)

        if keep_shape:
            C_new = x.shape[-1]
            H_new = H // self.patch_embed.patch_size[0]
            W_new = W // self.patch_embed.patch_size[1]
            x = x.transpose(1,2).contiguous().reshape(B, C_new, H_new, W_new)

        return x


def vit_xtiny_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=(16, 16), embed_dim=160, depth=8, num_heads=4, mlp_ratio=4, 
        qkv_bias=True, **kwargs)
    return model

def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=(16, 16), embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, 
        qkv_bias=True, **kwargs)
    return model

def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=(16, 16), embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, 
        qkv_bias=True, **kwargs)
    return model

def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=(14,14), embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, 
        qkv_bias=True, **kwargs)
    return model