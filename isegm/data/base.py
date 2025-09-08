import random
import pickle
from collections import namedtuple

import numpy as np
import torch
from torchvision import transforms
from .points_sampler import MultiPointSampler, MultiClassSampler
from .sample import DSample


class ISDataset(torch.utils.data.dataset.Dataset):
    def __init__(self,
                 augmentator=None,
                 points_sampler=MultiPointSampler(max_num_points=12),
                 min_object_area=0,
                 keep_background_prob=0.0,
                 with_image_info=False,
                 samples_scores_path=None,
                 samples_scores_gamma=1.0,
                 epoch_len=-1):
        super(ISDataset, self).__init__()
        self.epoch_len = epoch_len
        self.augmentator = augmentator
        self.min_object_area = min_object_area
        self.keep_background_prob = keep_background_prob
        self.points_sampler = points_sampler
        self.with_image_info = with_image_info
        self.samples_precomputed_scores = self._load_samples_scores(samples_scores_path, samples_scores_gamma)
        self.to_tensor = transforms.ToTensor()

        self.dataset_samples = None

    def __getitem__(self, index):
        '''

        Args:
            index:

        Returns:
            {
                'images': torch.Tensor, # The image tensor,
                'points': np.ndarray, # Points, take max_num_points as 24, then shape is (48, 3). First 24 is pos, last 24 is neg. First few is [y, x, 100], then extended with (-1, -1, -1).
                'instances': np.ndarray # The mask
            }

        '''
        if self.samples_precomputed_scores is not None:
            index = np.random.choice(self.samples_precomputed_scores['indices'],
                                     p=self.samples_precomputed_scores['probs'])
        else:
            if self.epoch_len > 0:
                index = random.randrange(0, len(self.dataset_samples))

        sample = self.get_sample(index)
        sample = self.augment_sample(sample)
        sample.remove_small_objects(self.min_object_area)

        self.points_sampler.sample_object(sample)
        points = np.array(self.points_sampler.sample_points())
        mask = self.points_sampler.selected_mask

        output = {
            'images': self.to_tensor(sample.image),
            'points': points.astype(np.float32),
            'instances': mask
        }

        if self.with_image_info:
            output['image_info'] = sample.sample_id

        return output

    def augment_sample(self, sample) -> DSample:
        if self.augmentator is None:
            return sample

        valid_augmentation = False
        while not valid_augmentation:
            sample.augment(self.augmentator)
            keep_sample = (self.keep_background_prob < 0.0 or
                           random.random() < self.keep_background_prob)
            valid_augmentation = len(sample) > 0 or keep_sample

        return sample

    def get_sample(self, index) -> DSample:
        raise NotImplementedError

    def __len__(self):
        if self.epoch_len > 0:
            return self.epoch_len
        else:
            return self.get_samples_number()

    def get_samples_number(self):
        return len(self.dataset_samples)

    @staticmethod
    def _load_samples_scores(samples_scores_path, samples_scores_gamma):
        if samples_scores_path is None:
            return None

        with open(samples_scores_path, 'rb') as f:
            images_scores = pickle.load(f)

        probs = np.array([(1.0 - x[2]) ** samples_scores_gamma for x in images_scores])
        probs /= probs.sum()
        samples_scores = {
            'indices': [x[0] for x in images_scores],
            'probs': probs
        }
        print(f'Loaded {len(probs)} weights with gamma={samples_scores_gamma}')
        return samples_scores

def is_dataset_collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    images = [d['images'] for d in data]
    points = [d['points'] for d in data]
    instances = [d['instances'] for d in data]

    all_points = points
    max_len = max([len(x) for x in all_points])
    for i in range(len(data)):
        padding_length = max_len - len(points[i])
        if padding_length > 0:
            # Create padding of shape (padding_length, 3) and fill it with (-1, -1, -1)
            padding = np.full((padding_length, 3), (-1, -1, -1))
            # Concatenate the original data with the padding
            padded_point = np.concatenate((all_points[i], padding), axis=0)
            all_points[i]=padded_point
    images = torch.stack(images)
    all_points = np.array(all_points)
    instances = np.array(instances)
    return images, torch.from_numpy(all_points), torch.from_numpy(instances)


