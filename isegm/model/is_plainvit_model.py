import math
import torch.nn as nn
from isegm.utils.serialization import serialize
from .is_model import ISModel
from .modeling.models_vit import VisionTransformer, PatchEmbed
from .modeling.swin_transformer import SwinTransfomerSegHead


class SimpleFPN(nn.Module):
    def __init__(self, in_dim=768, out_dims=[128, 256, 512, 1024]):
        super().__init__()
        self.down_4 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dims[0], 4, stride=4),
            nn.GroupNorm(1, out_dims[0]),
            nn.GELU()
        )
        self.down_8 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dims[1], 2, stride=2),
            nn.GroupNorm(1, out_dims[1]),
            nn.GELU()
        )
        self.down_16 = nn.Sequential(
            nn.Conv2d(in_dim, out_dims[2], 3, padding=1, stride=1),
            nn.GroupNorm(1, out_dims[2]),
            nn.GELU()
        )
        self.down_32 = nn.Sequential(
            nn.Conv2d(in_dim, out_dims[3], 3, padding=1, stride=2),
            nn.GroupNorm(1, out_dims[3]),
            nn.GELU()
        )

    def forward(self, x):
        x_down_4 = self.down_4(x)
        x_down_8 = self.down_8(x)
        x_down_16 = self.down_16(x)
        x_down_32 = self.down_32(x)

        return [x_down_4, x_down_8, x_down_16, x_down_32]


class PlainVitModel(ISModel):
    @serialize
    def __init__(
        self,
        backbone_params={},
        head_params={},
        neck_params={}, 
        **kwargs
        ):

        super().__init__(**kwargs)

        self.patch_embed_coords = PatchEmbed(
            img_size= backbone_params['img_size'],
            patch_size=backbone_params['patch_size'], 
            in_chans=3 if self.with_prev_mask else 2, 
            embed_dim=backbone_params['embed_dim'],
        )

        self.backbone = VisionTransformer(**backbone_params)
        self.neck = SimpleFPN(**neck_params)
        self.head = SwinTransfomerSegHead(**head_params)

    def backbone_forward(self, image, coord_features=None):
        coord_features = self.patch_embed_coords(coord_features)
        backbone_features = self.backbone.forward_backbone(image, coord_features)[:, 1:]

        # Extract 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        B, N, C = backbone_features.shape
        M = int(math.sqrt(N))
        assert M * M == N

        backbone_features = backbone_features.transpose(-1,-2).view(B, C, M, M)
        multi_scale_features = self.neck(backbone_features)

        return {'instances': self.head(multi_scale_features), 'instances_aux': None}
