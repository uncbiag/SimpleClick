import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from isegm.model.is_plainvit_model import PlainVitModel
from isegm.inference.clicker import Clicker
from isegm.inference.transform import ResizeLongestSide


class BasePredictor(object):
    def __init__(self, model: PlainVitModel, target_length: int=672) -> None:
        """
        Uses PlainViTModel to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          model: The SimpleClick model for mask prediction.
        """
        super().__init__()                
        self.model = model
        self.target_length = target_length
        self.to_tensor = transforms.ToTensor()
        self.transform = ResizeLongestSide(target_length)

    def set_image(self, image: np.ndarray) -> None:
        """TBD"""
        image = self.to_tensor(image).to(self.device)
        if len(image.shape) == 3:
            image = image.unsqueeze(0) # CHW -> BCHW
        self.orig_img_shape = image.shape
        
        image = self.preprocess_image(image)
        self.image_feats = self.model.get_image_feats(image)
        self.prev_mask = torch.zeros_like(image[:, :1, :, :])

    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """Resize image and pad to a square"""
        # Resize
        image = self.transform.apply_image_torch(image)

        # Pad to square
        h, w = image.shape[-2:]
        padh = self.target_length - h
        padw = self.target_length - w
        image = F.pad(image, (0, padw, 0, padh))
        return image

    def postprocess_mask(self, mask: torch.Tensor) -> torch.Tensor:
        # unpad
        orig_h, orig_w = self.orig_img_shape[-2:]
        mask = mask[..., :orig_h, :orig_w]

        # resize 
        mask = F.interpolate(mask, (orig_h, orig_w), mode='bilinear', align_corners=False)
        return mask

    def predict(self, clicker: Clicker) -> np.ndarray:
        """
        TBD
        """
        clicks_list = clicker.get_clicks()
        points_nd = self.get_points_nd([clicks_list])
        points_nd = self.transform.apply_coords_torch(points_nd)

        prompts = {'points': points_nd, 'prev_mask': self.prev_mask}
        prompt_feats = self.model.get_prompt_feats(self.image.shape, prompts)
        pred_logits = self.model(self.image.shape, self.image_feats, prompt_feats)['instances']
        pred_logits = self.postprocess_mask(pred_logits)

        prediction = torch.sigmoid(pred_logits)
        self.prev_mask = prediction

        return prediction.cpu().numpy()[0, 0]

    def get_points_nd(self, clicks_lists) -> torch.Tensor:
        total_clicks = []
        num_pos_clicks = [sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists]
        num_neg_clicks = [len(clicks_list) - num_pos for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)]
        num_max_points = max(num_pos_clicks + num_neg_clicks)
        num_max_points = max(1, num_max_points)

        for clicks_list in clicks_lists:
            pos_clicks = [click.coords_and_indx for click in clicks_list if click.is_positive]
            pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]

            neg_clicks = [click.coords_and_indx for click in clicks_list if not click.is_positive]
            neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
            total_clicks.append(pos_clicks + neg_clicks)

        return torch.tensor(total_clicks, device=self.device)
    
    @property
    def device(self) -> torch.device:
        return self.model.device