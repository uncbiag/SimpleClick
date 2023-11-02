from typing import Any, Dict
import cv2
import random

from albumentations.core.serialization import SERIALIZABLE_REGISTRY
from albumentations import ImageOnlyTransform, DualTransform
from albumentations.augmentations.geometric import functional as F


class ResizeLongestSide(DualTransform):
    """
    Resize images to the longest side 'target_length'.
    """
    def __init__(
        self, 
        target_length, 
        interpolation=cv2.INTER_LINEAR, 
        always_apply=False, 
        p=1
    ) -> None:
        super().__init__(always_apply, p)
        self.target_length = target_length
        self.interpolation = interpolation

    def get_params_dependent_on_targets(
        self, 
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        height, width = params['image'].shape[0], params['image'].shape[1]
        scale = self.target_length * 1.0 / max(height, width)
        if height > width:
            height = self.target_length
            width = int(round(width * scale))
        else:
            height = int(round(height * scale))
            width = self.target_length
        return {'new_height': height, 'new_width': width}

    def apply(
        self,
        img,
        new_height=0,
        new_width=0,
        interpolation=cv2.INTER_LINEAR,
        **params,
    ):
        return F.resize(img, height=new_height, width=new_width, 
                        interpolation=interpolation)

    def apply_to_keypoint(self, keypoint, new_height=0, new_width=0, **params):
        scale_x = new_width / params["cols"]
        scale_y = new_height / params["rows"]
        return F.keypoint_scale(keypoint, scale_x, scale_y)

    def get_transform_init_args_names(self):
        return "target_length", "interpolation"

    @property
    def targets_as_params(self):
        return ["image"]    


class UniformRandomResize(DualTransform):
    def __init__(
        self, 
        scale_range=(0.9, 1.1), 
        interpolation=cv2.INTER_LINEAR, 
        always_apply=False, 
        p=1
    ) -> None:
        super().__init__(always_apply, p)
        self.scale_range = scale_range
        self.interpolation = interpolation

    def get_params_dependent_on_targets(self, params):
        scale = random.uniform(*self.scale_range)
        height = int(round(params['image'].shape[0] * scale))
        width = int(round(params['image'].shape[1] * scale))
        return {'new_height': height, 'new_width': width}

    def apply(
        self, 
        img, 
        new_height=0, 
        new_width=0, 
        interpolation=cv2.INTER_LINEAR, 
        **params,
    ):
        return F.resize(img, height=new_height, width=new_width, 
                        interpolation=interpolation)

    def apply_to_keypoint(self, keypoint, new_height=0, new_width=0, **params):
        scale_x = new_width / params["cols"]
        scale_y = new_height / params["rows"]
        return F.keypoint_scale(keypoint, scale_x, scale_y)

    def get_transform_init_args_names(self):
        return "scale_range", "interpolation"

    @property
    def targets_as_params(self):
        return ["image"]
    

def remove_image_only_transforms(sdict):
    if not 'transforms' in sdict:
        return sdict

    keep_transforms = []
    for tdict in sdict['transforms']:
        cls = SERIALIZABLE_REGISTRY[tdict['__class_fullname__']]
        if 'transforms' in tdict:
            keep_transforms.append(remove_image_only_transforms(tdict))
        elif not issubclass(cls, ImageOnlyTransform):
            keep_transforms.append(tdict)
    sdict['transforms'] = keep_transforms

    return sdict