from isegm.inference.clicker import Clicker, Click
from isegm.model.is_plainvit_model import MultiOutVitModel
from isegm.utils.exp_imports.default import *
from isegm.model.modeling.transformer_helper.cross_entropy_loss import CrossEntropyLoss

def init_model():
    model_cfg = edict()
    model_cfg.crop_size = (448, 448)
    model_cfg.num_max_points = 24

    backbone_params = dict(
        img_size=model_cfg.crop_size,
        patch_size=(16,16),
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
    )

    neck_params = dict(
        in_dim = 1024,
        out_dims = [192, 384, 768, 1536],
    )

    head_params = dict(
        in_channels=[192, 384, 768, 1536],
        in_index=[0, 1, 2, 3],
        dropout_ratio=0.1,
        num_classes=7,
        loss_decode=CrossEntropyLoss(),
        align_corners=False,
        upsample='x1',
        channels={'x1': 256, 'x2': 128, 'x4': 64}['x1'],
    )

    model = MultiOutVitModel(
        use_disks=True,
        norm_radius=5,
        with_prev_mask=True,
        backbone_params=backbone_params,
        neck_params=neck_params,
        head_params=head_params,
        random_split=False,
    )

    # model.backbone.init_weights_from_pretrained("./weights/pretrained/cocolvis_vit_huge.pth")
    model.to("cuda")

    return model, model_cfg

def get_points_nd(clicks_lists):
    total_clicks = []
    num_pos_clicks = [sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists]
    num_neg_clicks = [len(clicks_list) - num_pos for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)]
    num_max_points = max(num_pos_clicks + num_neg_clicks)
    num_max_points = max(1, num_max_points)

    for clicks_list in clicks_lists:
        clicks_list = clicks_list[:5]
        pos_clicks = [click.coords_and_indx for click in clicks_list if click.is_positive]
        pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]

        neg_clicks = [click.coords_and_indx for click in clicks_list if not click.is_positive]
        neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
        total_clicks.append(pos_clicks + neg_clicks)

    return torch.tensor(total_clicks, device="cuda")

def add_mask(img):
    input_image = torch.cat((img, torch.zeros(1,1,448,448).cuda()), dim=1)
    return input_image


model, model_cfg = init_model()

import torch
import cv2
import numpy as np

img = torch.rand(1, 3, 448, 448).cuda()
click = Click(is_positive=True, coords=(1, 1), indx=0)
click_list = [[click]]
out = model(add_mask(img), get_points_nd(click_list))

print("done!")