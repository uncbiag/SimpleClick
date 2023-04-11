import os
import random
import pickle as pkl
from pathlib import Path

import cv2
import numpy as np

from isegm.data.base import ISDataset
from isegm.data.sample import DSample
from isegm.utils.misc import get_labels_with_sizes


def get_labelsdataset():
    labels = []
    path = os.path.dirname(os.path.abspath(__file__)) + \
        '/label_files/ade20k_objectInfo150.txt'
    assert os.path.exists(
         path), '*** Error : {} not exist !!!'.format(path)
    f = open(path, 'r')
    lines = f.readlines()
    for line in lines:
        label = line.strip().split(',')[-1].split(';')[0]
        labels.append(label)
    f.close()
    return labels


class ADE20kDataset(ISDataset):
    """ The object ids range from 1 to 150.
    """
    def __init__(self, dataset_path, images_dir_name='images', masks_dir_name='annotations',
                 split='train', **kwargs):
        super().__init__(**kwargs)
        assert split in {'train', 'val'}

        self.dataset_path = Path(dataset_path)
        self.dataset_split = split
        self.dataset_split_folder = 'training' if split == 'train' else 'validation'

        self._images_path = self.dataset_path / images_dir_name / self.dataset_split_folder
        self._insts_path = self.dataset_path / masks_dir_name / self.dataset_split_folder

        self.dataset_samples = [x.name for x in sorted(self._images_path.glob('*.*'))]
        self._masks_paths = {x.stem: x for x in self._insts_path.glob('*.*')}

        self.category_names = get_labelsdataset()

    def get_sample(self, index) -> DSample:
        image_id = self.dataset_samples[index]
        image_path = str(self._images_path / image_id)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = str(self._masks_paths[image_id.split('.')[0]])
        instances_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)      
        instances_mask = instances_mask.astype(np.int32)
        object_ids, _ = get_labels_with_sizes(instances_mask)

        objects_category_names = [self.category_names[idx] for idx in object_ids]

        return DSample(image, instances_mask, objects_ids=object_ids, sample_id=index,
                       objects_category_names=objects_category_names)