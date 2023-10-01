import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from isegm.model.ops import DistMaps, BatchImageNormalize
from isegm.model.modeling.models_vit import VisionTransformer, PatchEmbed, Block
from isegm.model.modeling.cross_attention import CrossBlock
from isegm.utils.serialization import serialize


class SimpleFPN(nn.Module):
    def __init__(self, in_dim=768, out_dims=[128, 256, 512, 1024]):
        super().__init__()
        self.down_4_chan = max(out_dims[0]*2, in_dim // 2)
        self.down_4 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, self.down_4_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_4_chan),
            nn.GELU(),
            nn.ConvTranspose2d(self.down_4_chan, self.down_4_chan // 2, 2, stride=2),
            nn.GroupNorm(1, self.down_4_chan // 2),
            nn.Conv2d(self.down_4_chan // 2, out_dims[0], 1),
            nn.GroupNorm(1, out_dims[0]),
            nn.GELU()
        )
        self.down_8_chan = max(out_dims[1], in_dim // 2)
        self.down_8 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, self.down_8_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_8_chan),
            nn.Conv2d(self.down_8_chan, out_dims[1], 1),
            nn.GroupNorm(1, out_dims[1]),
            nn.GELU()
        )
        self.down_16 = nn.Sequential(
            nn.Conv2d(in_dim, out_dims[2], 1),
            nn.GroupNorm(1, out_dims[2]),
            nn.GELU()
        )
        self.down_32_chan = max(out_dims[3], in_dim * 2)
        self.down_32 = nn.Sequential(
            nn.Conv2d(in_dim, self.down_32_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_32_chan),
            nn.Conv2d(self.down_32_chan, out_dims[3], 1),
            nn.GroupNorm(1, out_dims[3]),
            nn.GELU()
        )

        self.init_weights()

    def init_weights(self):
        # TODO
        pass

    def forward(self, x):
        x_down_4 = self.down_4(x)
        x_down_8 = self.down_8(x)
        x_down_16 = self.down_16(x)
        x_down_32 = self.down_32(x)

        return [x_down_4, x_down_8, x_down_16, x_down_32]


class SegmentationHead(nn.Module):
    """ The all MLP segmentation head
    """
    def __init__(self, in_select_index, in_channels, out_channels, dropout_ratio, 
                 num_classes, interpolate_mode='bilinear', align_corners=False):
        super().__init__()

        self.in_select_index=in_select_index
        self.dropout_ratio = dropout_ratio
        self.interpolate_mode = interpolate_mode
        self.align_corners = align_corners

        assert len(in_channels) == len(in_select_index)
        num_inputs = len(in_channels)

        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=in_channels[i],
                    out_channels=out_channels,
                    kernel_size=1))

        self.fusion_conv = ConvModule(
            in_channels=out_channels * num_inputs,
            out_channels=out_channels,
            kernel_size=1)

        self.seg_conv = nn.Conv2d(out_channels, num_classes, kernel_size=1)

    def forward(self, inputs):
        inputs = [inputs[i] for i in self.in_select_index]
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                F.interpolate(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.seg_conv(out)

        return out


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., mlp_drop=0., qkv_bias=False,
            attn_drop=0., proj_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.image_to_prompt = CrossBlock(dim=dim, num_heads=num_heads,
            mlp_ratio=mlp_ratio, mlp_drop=mlp_drop, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=proj_drop, act_layer=act_layer, 
            norm_layer=norm_layer)
        
        self.prompt_to_image = CrossBlock(dim=dim, num_heads=num_heads,
            mlp_ratio=mlp_ratio, mlp_drop=mlp_drop, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=proj_drop, act_layer=act_layer, 
            norm_layer=norm_layer)


    def forward(self, image_feats, prompt_feats, keep_shape=False):
        """ image_feats: tensor of shape [B, C, H, W]
            prompt_feats: tensor of shape [B, C, H, W]
        """
        assert image_feats.shape == prompt_feats.shape

        # reshape to [B, N, C]
        B, C, H, W = image_feats.shape
        N = H * W

        prompt_feats = prompt_feats.permute(0, 2, 3, 1).contiguous().reshape(B, N, C)
        image_feats = image_feats.permute(0, 2, 3, 1).contiguous().reshape(B, N, C)

        prompt_feats = self.prompt_to_image(prompt_feats, image_feats)
        image_feats = self.image_to_prompt(image_feats, prompt_feats)

        if keep_shape:
            prompt_feats = prompt_feats.permute(0, 2, 1)
            prompt_feats = prompt_feats.contiguous().reshape(B, C, H, W)

            image_feats = image_feats.permute(0, 2, 1)
            image_feats = image_feats.contiguous().reshape(B, C, H, W)

        return image_feats, prompt_feats


class PlainVitModel(nn.Module):
    @serialize
    def __init__(
        self,
        backbone_params={},
        neck_params={}, 
        head_params={},
        fusion_params={},
        with_aux_output=False, 
        norm_radius=5, 
        use_disks=False, 
        cpu_dist_maps=False, 
        with_prev_mask=False, 
        norm_mean_std=([.485, .456, .406], [.229, .224, .225])        
        ):

        super().__init__()

        self.with_aux_output = with_aux_output
        self.with_prev_mask = with_prev_mask
        self.normalization = BatchImageNormalize(norm_mean_std[0], norm_mean_std[1])

        self.coord_feature_ch = 2
        if self.with_prev_mask:
            self.coord_feature_ch += 1

        self.dist_maps = DistMaps(
            norm_radius=norm_radius, 
            spatial_scale=1.0,
            cpu_mode=cpu_dist_maps, 
            use_disks=use_disks
        )

        self.prompts_patch_embed = PatchEmbed(
            img_size= backbone_params['img_size'],
            patch_size=backbone_params['patch_size'], 
            in_chans=3 if self.with_prev_mask else 2, 
            embed_dim=backbone_params['embed_dim'],
            flatten=True
        )

        self.backbone = VisionTransformer(**backbone_params)
        self.neck = SimpleFPN(**neck_params)
        self.head = SegmentationHead(**head_params)

        self.fusion_type = fusion_params['type']

        if self.fusion_type == 'cross_attention':
            depth = int(fusion_params['depth'])
            self.fusion_blocks = nn.Sequential(*[
                CrossAttentionBlock(**fusion_params['params'])
                for _ in range(depth)])
            
        elif self.fusion_type == 'self_attention':
            depth = int(fusion_params['depth'])
            self.fusion_blocks = nn.Sequential(*[
                Block(**fusion_params['params'])
                for _ in range(depth)])

    def get_image_feats(self, image, keep_shape=True):
        image = self.normalization(image)
        image_feats = self.backbone(image, keep_shape=keep_shape)

        return image_feats

    def get_prompt_feats(self, image_shape, prompts, keep_shape=True):
        points = prompts['points']
        points_maps = self.dist_maps(image_shape, points)

        # TODO: support more visual prompts such as scribbles, 
        # bounding boxes, and masks

        prev_mask = prompts['prev_mask']
        prompt_maps = torch.cat((prev_mask, points_maps), dim=1) 

        prompt_feats = self.prompts_patch_embed(prompt_maps)

        if keep_shape:
            B = image_shape[0]
            C_new = prompt_feats.shape[-1]
            H_new = image_shape[2] // self.prompts_patch_embed.patch_size[0]
            W_new = image_shape[3] // self.prompts_patch_embed.patch_size[1]

            prompt_feats = prompt_feats.transpose(1,2).contiguous()
            prompt_feats = prompt_feats.reshape(B, C_new, H_new, W_new)

        return prompt_feats

    def fusion(self, image_feats, prompt_feats):
        if self.fusion_type == 'naive':
            return image_feats + prompt_feats
        
        elif self.fusion_type == 'cross_attention':
            num_blocks = len(self.fusion_blocks)
            for i in range(num_blocks):
                image_feats, prompt_feats = self.fusion_blocks[i](
                    image_feats, prompt_feats, keep_shape=True)
            return image_feats
        
        elif self.fusion_type == 'self_attention':
            image_feats = image_feats + prompt_feats
            B, C, H, W = image_feats.shape
            image_feats = image_feats.permute(0, 2, 3, 1).contiguous().reshape(B, H*W, C)

            num_blocks = len(self.fusion_blocks)
            for i in range(num_blocks):
                image_feats = self.fusion_blocks[i](image_feats)            
            image_feats = image_feats.transpose(1, 2).contiguous().reshape(B, C, H, W)
            return image_feats

        else:
            raise ValueError('fusion type not defined')

    def forward(self, image_shape, image_feats, prompt_feats):
        fused_features = self.fusion(image_feats, prompt_feats)
        multiscale_features = self.neck(fused_features)
        seg_prob = self.head(multiscale_features)

        # TODO: remove padded area

        seg_prob = nn.functional.interpolate(
            seg_prob, 
            size=image_shape[2:], 
            mode='bilinear', 
            align_corners=True
        )

        return {'instances': seg_prob, 'instances_aux': None}
    
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device