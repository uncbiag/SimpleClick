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
        norm_radius=5, 
        use_disks=False, 
        cpu_dist_maps=False, 
        norm_mean_std=([.485, .456, .406], [.229, .224, .225])        
    ) -> None:
        super().__init__()
        self.normalization = BatchImageNormalize(norm_mean_std[0], norm_mean_std[1])

        self.dist_maps = DistMaps(
            norm_radius=norm_radius, 
            spatial_scale=1.0,
            cpu_mode=cpu_dist_maps, 
            use_disks=use_disks
        )

        self.backbone = VisionTransformer(**backbone_params)
        self.neck = SimpleFPN(**neck_params)
        self.head = SegmentationHead(**head_params)

        self.visual_prompts_encoder = PatchEmbed(
            img_size=backbone_params['img_size'],
            patch_size=backbone_params['patch_size'], 
            in_chans=3, # prev mask + pos & net clicks
            embed_dim=backbone_params['embed_dim'],
            flatten=True
        )

        self.fusion_type = fusion_params['type']
        if self.fusion_type == 'cross_attention': # still has bug???
            depth = int(fusion_params['depth'])
            self.fusion_blocks = nn.Sequential(*[
                CrossAttentionBlock(**fusion_params['params'])
                for _ in range(depth)])
        elif self.fusion_type == 'self_attention':
            depth = int(fusion_params['depth'])
            self.fusion_blocks = nn.Sequential(*[
                Block(**fusion_params['params'])
                for _ in range(depth)])

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize, resize, and pad the input image

        Arguments:
            x: input tensor of shape [B, C, H, W]
        """
        # normalize image
        x = self.normalization(x)

        # resize the longest side
        self.orig_size = oldh, oldw = x.shape[-2:]
        target_length = self.backbone.patch_embed.img_size[0]
        scale = target_length * 1.0 / max(oldh, oldw)
        newh, neww = int(oldh * scale + 0.5), int(oldw * scale + 0.5)        
        x = F.interpolate(x, (newh, neww), mode="bilinear", align_corners=False)

        # pad to square
        self.input_size = x.shape[-2:]
        padh, padw = target_length - newh, target_length - neww
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def postprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unpad and resize to original size
        """
        # unpad
        input_h, input_w = self.input_size
        x = x[..., :input_h, :input_w]

        # resize
        x = F.interpolate(x, self.orig_size, mode='bilinear', align_corners=False)
        return x

    def get_image_feats(self, image, keep_shape=True):
        image = self.preprocess(image)
        image_feats = self.backbone(image, keep_shape=keep_shape)
        return image_feats

    def get_prompt_feats(self, image_shape, prompts, keep_shape=True):
        """Get feature representation for prompts
        
        Arguments:
            image_shape: original image shape
            prompts: a dictionary containing all possible prompts

        Returns:
            prompt features
        """

        prev_mask = prompts['prev_mask']
        # resize the longest side
        prev_mask = F.interpolate(prev_mask, self.input_size, mode='bilinear', align_corners=False)

        # pad
        target_length = self.backbone.patch_embed.img_size[0]
        h, w = prev_mask.shape[-2:]
        padh, padw = target_length - h, target_length - w
        prev_mask = F.pad(prev_mask, (0, padw, 0, padh))

        points = prompts['points']
        # transform coords for image resize
        for batch_id in range(len(points)):
            for point_id in range(len(points[batch_id])):
                if points[batch_id, point_id, 2] > -1:
                    w, h = points[batch_id, point_id, 0], points[batch_id, point_id, 1]
                    w = int(w * (self.input_size[0] / self.orig_size[0]) + 0.5)
                    h = int(h * (self.input_size[1] / self.orig_size[1]) + 0.5)
                    points[batch_id, point_id, 0], points[batch_id, point_id, 1] = w, h 
        point_mask = self.dist_maps(prev_mask.shape, points)

        prompt_mask = torch.cat((prev_mask, point_mask), dim=1)
        prompt_feats = self.visual_prompts_encoder(prompt_mask)
        if keep_shape:
            B = image_shape[0]
            C_new = prompt_feats.shape[-1]
            H_new = target_length // self.visual_prompts_encoder.patch_size[0]
            W_new = target_length // self.visual_prompts_encoder.patch_size[1]

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
        target_length = self.backbone.patch_embed.img_size[0]
        seg_prob = F.interpolate(
            seg_prob, 
            size=(target_length, target_length), 
            mode='bilinear', 
            align_corners=True
        )

        # post process
        seg_prob = self.postprocess(seg_prob)

        return {'instances': seg_prob}
    
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device