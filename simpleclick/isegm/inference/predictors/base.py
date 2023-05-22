import torch
import torch.nn.functional as F
from torchvision import transforms

from isegm.inference.transforms import AddHorizontalFlip, SigmoidForPred, \
    LimitLongestSide, ZoomIn


class BasePredictor(object):
    def __init__(self, model, device, zoomin_params, with_flip, max_size=None):
        
        self.net = model
        self.original_image = None
        self.device = device
        self.prev_prediction = None
        self.to_tensor = transforms.ToTensor()

        # transform parameters
        self.with_flip = with_flip
        self.zoom_in = None

        self.transforms = []
        if zoomin_params:
            self.zoom_in = ZoomIn(**zoomin_params)
            self.transforms.append(self.zoom_in)

        if max_size is not None:
            self.transforms.append(LimitLongestSide(max_size=max_size))

        self.transforms.append(SigmoidForPred())

        if with_flip:
            self.transforms.append(AddHorizontalFlip())

    def set_input_image(self, image):
        image_nd = self.to_tensor(image)

        for transform in self.transforms:
            transform.reset()

        self.original_image = image_nd.to(self.device)
        if len(self.original_image.shape) == 3:
            self.original_image = self.original_image.unsqueeze(0)
        
        self.prev_prediction = torch.zeros_like(self.original_image[:, :1, :, :])
        self.image_feats = None

    def get_prediction(self, clicker, prev_mask=None):
        clicks_list = clicker.get_clicks()

        if prev_mask is None:
            prev_mask = self.prev_prediction

        input_image = self.original_image
        if hasattr(self.net, 'with_prev_mask') and self.net.with_prev_mask:
            input_image = torch.cat((input_image, prev_mask), dim=1)

        image_nd, clicks_lists, _ = self.apply_transforms(input_image, [clicks_list])

        pred_logits = self._get_prediction(image_nd, clicks_lists)
        prediction = F.interpolate(pred_logits, mode='bilinear', align_corners=True,
                                   size=image_nd.size()[2:])

        for t in reversed(self.transforms):
            prediction = t.inv_transform(prediction)

        # what's the purpose of this???
        if self.zoom_in is not None and self.zoom_in.check_possible_recalculation():
            return self.get_prediction(clicker)

        self.prev_prediction = prediction
        return prediction.cpu().numpy()[0, 0]

    def _get_prediction(self, image_nd, clicks_lists):
        points_nd = self.get_points_nd(clicks_lists)

        prev_mask = None
        if hasattr(self.net, 'with_prev_mask') and self.net.with_prev_mask:
            prev_mask = image_nd[:, 3:, :, :]
            image_nd = image_nd[:, :3, :, :]

        prompts = {'points': points_nd, 'prev_mask': prev_mask}
        prompt_feats = self.net.get_prompt_feats(image_nd.shape, prompts)
        image_feats = self.net.get_image_feats(image_nd)

        return self.net(image_nd.shape, image_feats, prompt_feats)['instances']

    def apply_transforms(self, image_nd, clicks_lists):
        is_image_changed = False
        for t in self.transforms:
            image_nd, clicks_lists = t.transform(image_nd, clicks_lists)
            is_image_changed |= t.image_changed

        return image_nd, clicks_lists, is_image_changed

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