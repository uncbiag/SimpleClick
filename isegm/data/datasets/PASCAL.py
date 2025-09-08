import os
import pickle as pkl
import random
from pathlib import Path

import cv2
import numpy as np

from isegm.data.base import ISDataset
from isegm.data.sample import DSample
from isegm.utils.misc import get_bbox_from_mask, get_labels_with_sizes
from tqdm import tqdm
import pickle


class PASCAL(ISDataset):
    def __init__(self, dataset_path, split='train', **kwargs):
        super().__init__(**kwargs)
        assert split in {'train', 'val', 'trainval', 'test'}

        self._buggy_mask_thresh = 0.08
        self._buggy_objects = dict()

        self.name = 'PASCAL'
        self.dataset_path = Path(dataset_path)
        self._images_path = self.dataset_path / "JPEGImages"
        self._insts_path = self.dataset_path / "SegmentationPart_label"
        self.init_path = self.dataset_path / "voc_person_interactive_center_point"
        self.dataset_split = split
        self.class_num = 7 # 这个class_num 指所有在miou中可以被计算的类，包含背景类但不包含忽略区域
        self.ignore_id = 255

        self.loadfile = self.dataset_split+".pkl"
        if os.path.exists(str(self.dataset_path/"pascal_person_part_trainval_list"/self.loadfile)):
            with open(str(self.dataset_path/"pascal_person_part_trainval_list"/self.loadfile), 'rb') as file:
                self.dataset_samples = pickle.load(file)
        else:
            dataset_samples = []
            idsfile = self.dataset_split+"_id.txt"
            with open(str(self.dataset_path/"pascal_person_part_trainval_list"/idsfile), "r") as f:
                id_list = [line.strip() for line in f.readlines()]
            for id in id_list:
                img_path = self._images_path/(id+".jpg")
                gt_path = self._insts_path/(id+".png")
                init_path = self.init_path/(id+".png")
                dataset_samples.append((img_path, gt_path, init_path))
            image_id_lst = self.get_images_and_ids_list(dataset_samples)
            self.dataset_samples = image_id_lst
            # print(image_id_lst[:5])

    '''
    def get_sample(self, index) -> DSample:
        sample_id = self.dataset_samples[index]
        image_path = str(self._images_path / f'{sample_id}.jpg')
        mask_path = str(self._insts_path / f'{sample_id}.png')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = cv2.imread(mask_path)
        instances_mask = cv2.cvtColor(instances_mask, cv2.COLOR_BGR2GRAY).astype(np.int32)
        if self.dataset_split == 'test':
            instance_id = self.instance_ids[index]
            mask = np.zeros_like(instances_mask)
            mask[instances_mask == 220] = 220  # ignored area
            mask[instances_mask == instance_id] = 1
            objects_ids = [1]
            instances_mask = mask
        else:
            objects_ids = np.unique(instances_mask)
            objects_ids = [x for x in objects_ids if x != 0 and x != 220]

        return DSample(image, instances_mask, objects_ids=objects_ids, ignore_ids=[220], sample_id=index)
    '''

    def get_sample(self, index) -> DSample:
        sample_path, target_path, instance_ids, init_path = self.dataset_samples[index]
        # sample_id = str(sample_id)
        # print(sample_id)
        # num_zero = 6 - len(sample_id)
        # sample_id = '2007_'+'0'*num_zero + sample_id

        image_path = str(sample_path)
        mask_path = str(target_path)
        init_path = str(init_path)

        # print(image_path)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = cv2.imread(mask_path)
        instances_mask = cv2.cvtColor(instances_mask, cv2.COLOR_BGR2GRAY).astype(np.int32)
        # mask[instances_mask == 255] = 220  # ignored area
        # mask[instances_mask == instance_id] = 1
        objects_ids = instance_ids # 现在instance_ids 是一个列表
        return DSample(image, instances_mask, objects_ids=objects_ids, ignore_ids=[self.ignore_id], sample_id=index, init_clicks=init_path)

    def get_images_and_ids_list(self, dataset_samples, ignore_id = 255):
        images_and_ids_list = []
        object_count = 0
        # for i in tqdm(range(len(dataset_samples))):
        for i in range(len(dataset_samples)):
            image_path, mask_path, init_path = dataset_samples[i]
            instances_mask = cv2.imread(str(mask_path))
            instances_mask = cv2.cvtColor(instances_mask, cv2.COLOR_BGR2GRAY).astype(np.int32)
            objects_ids = np.unique(instances_mask)
    
            objects_ids = [x for x in objects_ids if x != ignore_id]
            object_count+=len(objects_ids)
            # for j in objects_ids:
            images_and_ids_list.append([image_path, mask_path ,objects_ids, init_path])
                # print(i,j,objects_ids)
        with open(str(self.dataset_path/"pascal_person_part_trainval_list"/self.loadfile), "wb") as file:
            pickle.dump(images_and_ids_list, file)
        print("file count: "+str(len(dataset_samples)))
        print("object count: "+str(object_count))
        return images_and_ids_list
    def remove_buggy_masks(self, index, instances_mask):
        if self._buggy_mask_thresh > 0.0:
            buggy_image_objects = self._buggy_objects.get(index, None)
            if buggy_image_objects is None:
                buggy_image_objects = []
                instances_ids, _ = get_labels_with_sizes(instances_mask)
                for obj_id in instances_ids:
                    obj_mask = instances_mask == obj_id
                    mask_area = obj_mask.sum()
                    bbox = get_bbox_from_mask(obj_mask)
                    bbox_area = (bbox[1] - bbox[0] + 1) * (bbox[3] - bbox[2] + 1)
                    obj_area_ratio = mask_area / bbox_area
                    if obj_area_ratio < self._buggy_mask_thresh:
                        buggy_image_objects.append(obj_id)

                self._buggy_objects[index] = buggy_image_objects
            for obj_id in buggy_image_objects:
                instances_mask[instances_mask == obj_id] = 0

        return instances_mask

    def __len__(self):
        return len(self.dataset_samples)

    def __getitem__(self, index): # points should be sampled from the whole mask
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
        init_points = cv2.imread(sample.init_clicks)[:,:,0]
        rows, cols = np.where(init_points != 255)
        non_255_values = init_points[rows, cols]
        coords_and_values = list(zip(rows, cols, non_255_values))
        coords_and_values.extend([(-1, -1, -1)] * (self.points_sampler.max_num_points - len(coords_and_values)))
        init_points = np.array(coords_and_values)
        self.points_sampler.sample_object(sample)
        points = np.array(self.points_sampler.sample_points())

        mask = self.points_sampler.selected_mask

        output = {
            'images': self.to_tensor(sample.image),
            'points': init_points.astype(np.float32),
            'instances': mask,
            # 'init_points': init_points.astype(np.float32)
        }

        if self.with_image_info:
            output['image_info'] = sample.sample_id

        return output
    

