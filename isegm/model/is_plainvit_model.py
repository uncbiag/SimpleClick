import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from isegm.utils.serialization import serialize
from isegm.model.is_model import ISModel
from isegm.model.modeling.models_vit import VisionTransformer, PatchEmbed


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


class PlainVitModel(ISModel):
    @serialize
    def __init__(
        self,
        backbone_params={},
        neck_params={}, 
        head_params={},
        **kwargs
        ):

        super().__init__(**kwargs)

        self.patch_embed_coords = PatchEmbed(
            img_size= backbone_params['img_size'],
            patch_size=backbone_params['patch_size'], 
            in_chans=3 if self.with_prev_mask else 2, 
            embed_dim=backbone_params['embed_dim'],
            flatten=True
        )

        self.backbone = VisionTransformer(**backbone_params)
        self.neck = SimpleFPN(**neck_params)
        self.head = SegmentationHead(**head_params)

    def backbone_forward(self, image, coord_features):
        coord_features = self.patch_embed_coords(coord_features)
        single_scale_features = self.backbone(image, coord_features, keep_shape=True)
        multi_scale_features = self.neck(single_scale_features)
        seg_prob = self.head(multi_scale_features)

        return {'instances': seg_prob, 'instances_aux': None}
