from pathlib import Path

import cv2
import numpy as np

from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class OAIDataset(ISDataset):
    def __init__(self, dataset_path, split='train',
                 images_dir_name='image', masks_dir_name='annotations',
                 **kwargs):
        super(OAIDataset, self).__init__(**kwargs)

        self.dataset_path = Path(dataset_path)
        self._images_path = self.dataset_path / split / images_dir_name
        self._insts_path = self.dataset_path / split / masks_dir_name

        self.dataset_samples = [x.name for x in sorted(self._images_path.glob('*.png'))]
        self._masks_paths = {x.stem: x for x in self._insts_path.glob('*.png')}

        assert len(self.dataset_samples) > 0

    def get_sample(self, index) -> DSample:
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / image_name)
        mask_path = str(self._masks_paths[image_name.split('.')[0]])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = cv2.imread(mask_path)[:, :, 0].astype(np.uint8)

        objects_ids = np.unique(instances_mask)
        objects_ids = [x for x in objects_ids]

        return DSample(image, instances_mask, objects_ids=objects_ids, ignore_ids=[-1], sample_id=index)
