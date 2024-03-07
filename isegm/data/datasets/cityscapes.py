import os
import pickle as pkl
from pathlib import Path
import random
import cv2
import numpy as np

from isegm.data.base import ISDataset
from isegm.data.sample import DSample
from isegm.utils.misc import get_bbox_from_mask, get_labels_with_sizes
from tqdm import tqdm
import pickle


class CityScapes(ISDataset):
    def __init__(self, dataset_path, split="train", **kwargs):
        super(CityScapes, self).__init__(**kwargs)
        assert split in {"train", "val", "trainval", "test"}
        self.name = "Cityscapes"
        self.dataset_path = Path(dataset_path)
        self._images_path = self.dataset_path / "leftImg8bit" / split
        self._insts_path = self.dataset_path / "gtFine" / split
        self.init_path = self.dataset_path / "init_interactive_point"
        self.dataset_split = split
        self.class_num = 19
        self.ignore_id = 255

        self.loadfile = self.dataset_split+".pkl"
        if os.path.exists(str(self.dataset_path/self.loadfile)):
            with open(str(self.dataset_path/self.loadfile), 'rb') as file:
                self.dataset_samples = pickle.load(file)
        else:
            dataset_samples = []
            for city in os.listdir(self._images_path):
                img_dir = self._images_path / city
                target_dir = self._insts_path / city
                init_dir = self.init_path / city
                for file_name in os.listdir(img_dir):
                    toAddPath = img_dir / file_name
                    initName = file_name.replace("_leftImg8bit", "")
                    initPath = init_dir / initName
                    labelName = file_name.replace("leftImg8bit", "gtFine_labelTrainIds")
                    labelPath = target_dir / labelName
                    dataset_samples.append((toAddPath, labelPath, initPath))
            
            image_id_lst = self.get_images_and_ids_list(dataset_samples)
            self.dataset_samples = image_id_lst
        # print(image_id_lst[:5])

    def get_sample(self, index) -> DSample:
        sample_path, target_path, instance_ids, init_path = self.dataset_samples[index]

        image_path = str(sample_path)
        mask_path = str(target_path)
        init_path = str(init_path)


        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = cv2.imread(mask_path)
        instances_mask = cv2.cvtColor(instances_mask, cv2.COLOR_BGR2GRAY).astype(
            np.int32
        )

        ids = [x for x in np.unique(instances_mask) if x != self.ignore_id]

        objects_ids = ids  # 现在instance_ids 是一个列表

        return DSample(
            image,
            instances_mask,
            objects_ids=objects_ids,
            ignore_ids=[self.ignore_id],
            sample_id=index,
            init_clicks=init_path,
        )

    def get_images_and_ids_list(self, dataset_samples):
        images_and_ids_list = []
        object_count = 0
        # for i in tqdm(range(len(dataset_samples))):
        for i in range(len(dataset_samples)):
            image_path, mask_path, init_path = dataset_samples[i]
            instances_mask = cv2.imread(str(mask_path))
            instances_mask = cv2.cvtColor(instances_mask, cv2.COLOR_BGR2GRAY).astype(
                np.int32
            )
            objects_ids = np.unique(instances_mask)

            objects_ids = [x for x in objects_ids if x != self.ignore_id]
            object_count += len(objects_ids)

            images_and_ids_list.append([image_path, mask_path, objects_ids, init_path])

        with open(str(self.dataset_path/self.loadfile), "wb") as file:
            pickle.dump(images_and_ids_list, file)
        return images_and_ids_list


class CityScapes_train(ISDataset):
    def __init__(self, dataset_path, split="train", **kwargs):
        super(CityScapes_train, self).__init__(**kwargs)
        assert split in {"train", "val", "trainval", "test"}

        self._buggy_mask_thresh = 0.08
        self._buggy_objects = dict()

        self.name = "Cityscapes"
        self.dataset_path = Path(dataset_path)
        self._images_path = self.dataset_path / "leftImg8bit" / split
        self._insts_path = self.dataset_path / "gtFine" / split
        self.dataset_split = split

        dataset_samples = []
        for city in os.listdir(self._images_path):
            img_dir = self._images_path / city
            target_dir = self._insts_path / city

            for file_name in os.listdir(img_dir):
                toAddPath = img_dir / file_name
                labelName = file_name.replace("leftImg8bit", "gtFine_labelTrainIds")
                labelPath = target_dir / labelName
                dataset_samples.append((toAddPath, labelPath))
        self.dataset_samples = dataset_samples
        # print(image_id_lst[:5])

    def get_sample(self, index) -> DSample:
        sample_path, target_path = self.dataset_samples[index]
        # sample_id = str(sample_id)
        # print(sample_id)
        # num_zero = 6 - len(sample_id)
        # sample_id = '2007_'+'0'*num_zero + sample_id

        image_path = str(sample_path)
        mask_path = str(target_path)

        # print(image_path)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = cv2.imread(mask_path)
        instances_mask = cv2.cvtColor(instances_mask, cv2.COLOR_BGR2GRAY).astype(
            np.int32
        )

        instances_mask = self.remove_buggy_masks(index, instances_mask)
        instances_ids, _ = get_labels_with_sizes(instances_mask, ignoreid=255)

        objects_ids = instances_ids

        return DSample(
            image,
            instances_mask,
            objects_ids=objects_ids,
            ignore_ids=[255],
            sample_id=index,
        )

    def get_images_and_ids_list(self, dataset_samples):
        images_and_ids_list = []
        # for i in tqdm(range(len(dataset_samples))):
        for i in range(len(dataset_samples)):
            image_path, mask_path = dataset_samples[i]
            instances_mask = cv2.imread(str(mask_path))
            instances_mask = cv2.cvtColor(instances_mask, cv2.COLOR_BGR2GRAY).astype(
                np.int32
            )
            objects_ids = np.unique(instances_mask)

            objects_ids = [x for x in objects_ids if x != 255]
            for j in objects_ids:
                images_and_ids_list.append([image_path, mask_path, j])
                # print(i,j,objects_ids)
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
