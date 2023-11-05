import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from isegm.model.is_plainvit_model import PlainVitModel
from isegm.inference.clicker import Clicker


class BasePredictor(object):
    def __init__(self, model: PlainVitModel) -> None:
        """
        Uses PlainViTModel to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          model: The SimpleClick model for mask prediction.
        """
        super().__init__()                
        self.model = model
        self.to_tensor = transforms.ToTensor()

    def set_image(self, image: np.ndarray) -> None:
        """Set image"""
        image = self.to_tensor(image).to(self.device)
        if len(image.shape) == 3:
            image = image.unsqueeze(0) # CHW -> BCHW
        self.orig_img_shape = image.shape
        self.prev_mask = torch.zeros_like(image, dtype=torch.float32)[:, :1, :, :]
        self.image_feats = self.model.get_image_feats(image)

    def predict(self, clicker: Clicker) -> np.ndarray:
        """
        TBD
        """
        clicks_list = clicker.get_clicks()
        points_nd = self.get_points_nd([clicks_list])        

        prompts = {'points': points_nd, 'prev_mask': self.prev_mask}
        prompt_feats = self.model.get_prompt_feats(self.orig_img_shape, prompts)

        pred_logits = self.model(self.orig_img_shape, self.image_feats, prompt_feats)['instances']
        prediction = torch.sigmoid(pred_logits)
        self.prev_mask = prediction

        return prediction.cpu().numpy()[0, 0]

    def get_points_nd(self, clicks_lists) -> torch.Tensor:
        """
        Arguments:
            clicks_lists: a list containing clicks list for a batch
            
        Returns:
            torch.Tensor: a tensor of points with shape B x 2N x 3 
        """
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
    
    def get_states(self):
        return {'prev_prediction': self.prev_mask.clone()}

    def set_states(self, states=None):
        self.prev_mask = states['prev_prediction']