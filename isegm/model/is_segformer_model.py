import torch.nn as nn

from isegm.utils.serialization import serialize
from .is_model import ISModel
from isegm.model.modifiers import LRMult
from .modeling.segformer import MixVisionTransformer, SegformerHead


class SegformerModel(ISModel):
    @serialize
    def __init__(
        self, 
        backbone_params=None, 
        decode_head_params=None,
        backbone_lr_mult=0.1, 
        **kwargs
        ):

        super().__init__(**kwargs)

        self.feature_extractor = MixVisionTransformer(**backbone_params)
        self.feature_extractor.apply(LRMult(backbone_lr_mult))

        self.head = SegformerHead(**decode_head_params)

    def backbone_forward(self, image, coord_features=None):
        backbone_features = self.feature_extractor(image, coord_features)
        return {'instances': self.head(backbone_features), 'instances_aux': None}
        