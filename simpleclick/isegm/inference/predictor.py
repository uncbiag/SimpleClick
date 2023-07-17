import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from isegm.model.is_plainvit_model import PlainVitModel
from isegm.inference.clicker import Clicker
from isegm.inference.transform import ResizeLongestSide


class BasePredictor(object):
    def __init__(
            self, 
            model: PlainVitModel,
        ) -> None:
        """
        Uses PlainViTModel to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          model: The SimpleClick model for mask prediction.
        """
        super().__init__()                
        self.model = model
        self.transform = ResizeLongestSide(1024)

    def set_image(
            self, 
            image: np.ndarray,
            image_format: str = "RGB",            
        ) -> None:
        """
        TBD
        """
        self.image = transforms.ToTensor()(image).to(self.device)
        if len(self.image.shape) == 3:
            # CHW -> BCHW
            self.image = self.image.unsqueeze(0)
        self.orig_h, self.orig_w = self.image.shape[2:]
        
        self.image = self.transform.apply_image_torch(self.image)
        self.image_feats = self.model.get_image_feats(self.image)
        self.prev_mask = torch.zeros_like(self.image[:, :1, :, :])

    def predict(
            self, 
            clicker: Clicker,
        ) -> np.ndarray:
        """
        TBD
        """
        clicks_list = clicker.get_clicks()
        points_nd = self.get_points_nd([clicks_list])

        prompts = {'points': points_nd, 'prev_mask': self.prev_mask}
        prompt_feats = self.model.get_prompt_feats(self.image.shape, prompts)
        pred_logits = self.model(self.image.shape, self.image_feats, prompt_feats)['instances']
 
        prediction = torch.sigmoid(pred_logits)
        self.prev_mask = prediction

        return prediction.cpu().numpy()[0, 0]

    def get_points_nd(self, clicks_lists):
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