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
    def __init__(self, dataset_path, split="train", use_cache=True, first_return_points="init", **kwargs):
        super(CityScapes, self).__init__(**kwargs)
        assert split in {"train", "val", "trainval", "test"}
        assert first_return_points in {"init", "random", "blank"}
        self.name = "Cityscapes"
        self.dataset_path = Path(dataset_path)
        self._images_path = Path("leftImg8bit") / split
        self._insts_path = Path("gtFine") / split
        self.init_path = Path("init_interactive_point")
        self.dataset_split = split
        self.class_num = 19
        self.ignore_id = 255
        self.first_return_points = first_return_points


        self.loadfile = self.dataset_split+".pkl"
        if os.path.exists(str(self.dataset_path/self.loadfile)) and use_cache:
            with open(str(self.dataset_path/self.loadfile), 'rb') as file:
                self.dataset_samples = pickle.load(file)
        else:
            dataset_samples = []
            for city in os.listdir(self.dataset_path/self._images_path):
                img_dir = self._images_path / city
                target_dir = self._insts_path / city
                init_dir = self.init_path / city
                for file_name in os.listdir(self.dataset_path/img_dir):
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

        image_path = str(self.dataset_path/sample_path)
        mask_path = str(self.dataset_path/target_path)
        init_path = str(self.dataset_path/init_path)


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
        for i in tqdm(range(len(dataset_samples))):
        # for i in range(len(dataset_samples)):
            image_path, mask_path, init_path = dataset_samples[i]
            instances_mask = cv2.imread(str(self.dataset_path/mask_path))
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

    def __len__(self):
        return len(self.dataset_samples)

    def __getitem__(self, index):
        sample = self.get_sample(index)
        sample = self.augment_sample(sample)
        init_points = cv2.imread(sample.init_clicks)[:, :, 0]
        rows, cols = np.where(init_points != 255)
        non_255_values = init_points[rows, cols]
        coords_and_values = list(zip(rows, cols, non_255_values))
        # coords_and_values.extend([(-1, -1, -1)] * (self.points_sampler.max_num_points - len(coords_and_values)))
        init_points = np.array(coords_and_values)
        self.points_sampler.sample_object(sample)
        mask = self.points_sampler.selected_mask
        if self.first_return_points=="init":
            points = init_points
        elif self.first_return_points=="random":
            points = np.array(self.points_sampler.sample_points())
        elif self.first_return_points=="blank":
            points = np.array([(-1, -1, -1)])
        output = {
            'images': self.to_tensor(sample.image),
            'points': points.astype(np.float32),
            'instances': mask,
            # 'init_points': init_points.astype(np.float32)
        }
        return output


