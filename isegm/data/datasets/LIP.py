import os
import pickle as pkl
from pathlib import Path

import cv2
import numpy as np

from isegm.data.base import ISDataset
from isegm.data.sample import DSample
from isegm.utils.misc import get_bbox_from_mask, get_labels_with_sizes

from tqdm import tqdm
import pickle


class LIP(ISDataset):
    def __init__(self, dataset_path, split='train', **kwargs):
        super().__init__(**kwargs)
        assert split in {'train', 'val', 'trainval', 'test'}
        self.name = 'LIP'
        self.dataset_path = Path(dataset_path)
        self._images_path = self.dataset_path / "train_images"
        self._insts_path = self.dataset_path / "train_segmentations"
        self.init_path = self.dataset_path / "20_cls_interactive_point"
        self.dataset_split = split
        self.class_num = 20 # 这个class_num 指所有在miou中可以被计算的类，包含背景类但不包含忽略区域
        self.ignore_id = 255

        self.loadfile = self.dataset_split+".pkl"
        if os.path.exists(str(self.dataset_path/self.loadfile)):
            with open(str(self.dataset_path/self.loadfile), 'rb') as file:
                self.dataset_samples = pickle.load(file)
        else:
            dataset_samples = []
            idsfile = self.dataset_split+"_id.txt"
            with open(str(self.dataset_path/idsfile), "r") as f:
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


        mask = instances_mask
        # mask[instances_mask == 255] = 220  # ignored area
        # mask[instances_mask == instance_id] = 1
        objects_ids = instance_ids # 现在instance_ids 是一个列表
        instances_mask = mask
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
        with open(str(self.dataset_path/self.loadfile), "wb") as file:
            pickle.dump(images_and_ids_list, file)
        print("file count: "+str(len(dataset_samples)))
        print("object count: "+str(object_count))
        return images_and_ids_list

class LIP_train(ISDataset):
    def __init__(self, dataset_path, split='train', **kwargs):
        super().__init__(**kwargs)
        assert split in {'train', 'val', 'trainval', 'test'}

        self._buggy_mask_thresh = 0.08
        self._buggy_objects = dict()

        self.name = 'LIP'
        self.dataset_path = Path(dataset_path)
        self._images_path = self.dataset_path / "train_images"
        self._insts_path = self.dataset_path / "train_segmentations"
        self.init_path = self.dataset_path / "20_cls_interactive_point"
        self.dataset_split = split
        self.class_num = 19 # 这个class_num 指所有在miou中可以被计算的类，包含背景类但不包含忽略区域
        self.ignore_id = 0

        self.loadfile = self.dataset_split+".pkl"
        if os.path.exists(str(self.dataset_path/self.loadfile)):
            with open(str(self.dataset_path/self.loadfile), 'rb') as file:
                self.dataset_samples = pickle.load(file)
        else:
            dataset_samples = []
            idsfile = self.dataset_split+"_id.txt"
            with open(str(self.dataset_path/idsfile), "r") as f:
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

        instances_mask = self.remove_buggy_masks(index, instances_mask)
        instances_ids, _ = get_labels_with_sizes(instances_mask, ignoreid=0)

        objects_ids = instances_ids # 现在instance_ids 是一个列表

        return DSample(image, instances_mask, objects_ids=objects_ids, ignore_ids=[0], sample_id=index)

    def get_images_and_ids_list(self, dataset_samples, ignore_id = 0):
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
        with open(str(self.dataset_path/self.loadfile), "wb") as file:
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