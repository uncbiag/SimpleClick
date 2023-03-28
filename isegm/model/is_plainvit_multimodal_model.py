
from isegm.utils.serialization import serialize
from isegm.model.is_model import MultiModalISModel
from isegm.model.modeling.models_vit import VisionTransformer, PatchEmbed
from isegm.model.is_plainvit_model import SimpleFPN, SegmentationHead


class MultiModalPlainVitModel(MultiModalISModel):
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

    def backbone_forward(self, image, coord_features, text_features=None, 
                         support_features=None):
        coord_features = self.patch_embed_coords(coord_features)

        single_scale_features = self.backbone(image, coord_features, keep_shape=True)
        multi_scale_features = self.neck(single_scale_features)
        seg_prob = self.head(multi_scale_features)

        return {'instances': seg_prob, 'instances_aux': None}
