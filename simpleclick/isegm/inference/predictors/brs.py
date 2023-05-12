import torch
import numpy as np

from .base import BasePredictor


class BRSBasePredictor(BasePredictor):
    def __init__(self, model, device, opt_functor, optimize_after_n_clicks=1, **kwargs):
        super().__init__(model, device, **kwargs)
        self.optimize_after_n_clicks = optimize_after_n_clicks
        self.opt_functor = opt_functor

        self.opt_data = None
        self.input_data = None

    def set_input_image(self, image):
        super().set_input_image(image)
        self.opt_data = None
        self.input_data = None

    def _get_clicks_maps_nd(self, clicks_lists, image_shape, radius=1):
        pos_clicks_map = np.zeros((len(clicks_lists), 1) + image_shape, dtype=np.float32)
        neg_clicks_map = np.zeros((len(clicks_lists), 1) + image_shape, dtype=np.float32)

        for list_indx, clicks_list in enumerate(clicks_lists):
            for click in clicks_list:
                y, x = click.coords
                y, x = int(round(y)), int(round(x))
                y1, x1 = y - radius, x - radius
                y2, x2 = y + radius + 1, x + radius + 1

                if click.is_positive:
                    pos_clicks_map[list_indx, 0, y1:y2, x1:x2] = True
                else:
                    neg_clicks_map[list_indx, 0, y1:y2, x1:x2] = True

        with torch.no_grad():
            pos_clicks_map = torch.from_numpy(pos_clicks_map).to(self.device)
            neg_clicks_map = torch.from_numpy(neg_clicks_map).to(self.device)

        return pos_clicks_map, neg_clicks_map

    def get_states(self):
        return {'transform_states': self._get_transform_states(), 'opt_data': self.opt_data}

    def set_states(self, states):
        self._set_transform_states(states['transform_states'])
        self.opt_data = states['opt_data']