import torch
import torch.nn as nn
import numpy as np

from isegm.model.ops import DistMaps, BatchImageNormalize


class ISModel(nn.Module):
    def __init__(self, with_aux_output=False, norm_radius=5, use_disks=False, 
                 cpu_dist_maps=False, with_prev_mask=False, 
                 norm_mean_std=([.485, .456, .406], [.229, .224, .225])):
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

    def forward(self, image, points):
        image, prev_mask = self.prepare_input(image)
        coord_features = self.get_coord_features(image, prev_mask, points)
        outputs = self.backbone_forward(image, coord_features)

        outputs['instances'] = nn.functional.interpolate(
            outputs['instances'], 
            size=image.size()[2:], 
            mode='bilinear', 
            align_corners=True
        )

        if self.with_aux_output:
            outputs['instances_aux'] = nn.functional.interpolate(
                outputs['instances_aux'], 
                size=image.size()[2:],
                mode='bilinear', 
                align_corners=True
            )

        return outputs

    def prepare_input(self, image):
        prev_mask = None
        if self.with_prev_mask:
            prev_mask = image[:, 3:, :, :]
            image = image[:, :3, :, :]

        image = self.normalization(image)
        return image, prev_mask

    def backbone_forward(self, image, coord_features):
        raise NotImplementedError

    def get_coord_features(self, image, prev_mask, points):
        coord_features = self.dist_maps(image, points)
        if prev_mask is not None:
            coord_features = torch.cat((prev_mask, coord_features), dim=1)

        return coord_features


def split_points_by_order(tpoints: torch.Tensor, groups):
    points = tpoints.cpu().numpy()
    num_groups = len(groups)
    bs = points.shape[0]
    num_points = points.shape[1] // 2

    groups = [x if x > 0 else num_points for x in groups]
    group_points = [np.full((bs, 2 * x, 3), -1, dtype=np.float32)
                    for x in groups]

    last_point_indx_group = np.zeros((bs, num_groups, 2), dtype=np.int)
    for group_indx, group_size in enumerate(groups):
        last_point_indx_group[:, group_indx, 1] = group_size

    for bindx in range(bs):
        for pindx in range(2 * num_points):
            point = points[bindx, pindx, :]
            group_id = int(point[2])
            if group_id < 0:
                continue

            is_negative = int(pindx >= num_points)
             # disable negative first click
            if group_id >= num_groups or (group_id == 0 and is_negative):
                group_id = num_groups - 1

            new_point_indx = last_point_indx_group[bindx, group_id, is_negative]
            last_point_indx_group[bindx, group_id, is_negative] += 1

            group_points[group_id][bindx, new_point_indx, :] = point

    group_points = [torch.tensor(x, dtype=tpoints.dtype, device=tpoints.device)
                    for x in group_points]

    return group_points