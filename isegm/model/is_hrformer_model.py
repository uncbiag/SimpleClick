import torch
import torch.nn as nn

from collections import OrderedDict

from isegm.utils.serialization import serialize
from .is_model import ISModel
from isegm.model.modifiers import LRMult
from .modeling.hrformer import HRT_B_OCR_V3

class HRFormerModel(ISModel):
    @serialize
    def __init__(
        self, 
        num_classes=1,
        in_ch=6,
        backbone_lr_mult=0.1, 
        **kwargs
        ):

        super().__init__(**kwargs)

        self.feature_extractor = HRT_B_OCR_V3(num_classes, in_ch)
        self.feature_extractor.apply(LRMult(backbone_lr_mult))

    def backbone_forward(self, image, coord_features=None):
        backbone_features = self.feature_extractor(image)
        return {'instances': backbone_features[0], 'instances_aux': backbone_features[1]}
        
    def init_weight(self, pretrained=None):
        if pretrained is not None:
            state_dict = torch.load(pretrained)['model']
            state_dict_rename = OrderedDict()
            for k, v in state_dict.items():
                state_dict_rename['backbone.' + k] = v

            ori_proj_weight = state_dict_rename['backbone.conv1.weight']
            state_dict_rename['backbone.conv1.weight'] = torch.cat([ori_proj_weight, ori_proj_weight], dim=1)

            self.feature_extractor.load_state_dict(state_dict_rename, False)
            print('Successfully loaded pretrained model.')
