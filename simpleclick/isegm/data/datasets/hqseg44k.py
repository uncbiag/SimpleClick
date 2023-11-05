from pathlib import Path

import cv2
import numpy as np

from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class HQSeg44kDataset(ISDataset):
    def __init__(
            self, 
            dataset_path,
            split='train',
            **kwargs
    ) -> None:
        super(HQSeg44kDataset, self).__init__(**kwargs)
        self.dataset_path = Path(dataset_path)

        # train set: 44320 images
        dis_train = {
            "name": "DIS5K-TR",
            "im_dir": "DIS5K/DIS-TR/im",
            "gt_dir": "DIS5K/DIS-TR/gt"}
        thin_train = {
            "name": "ThinObject5k-TR",
            "im_dir": "thin_object_detection/ThinObject5K/images_train",
            "gt_dir": "thin_object_detection/ThinObject5K/masks_train"}
        fss_train = {
            "name": "FSS",
            "im_dir": "cascade_psp/fss_all",
            "gt_dir": "cascade_psp/fss_all"}
        duts_train = {
            "name": "DUTS-TR",
            "im_dir": "cascade_psp/DUTS-TR",
            "gt_dir": "cascade_psp/DUTS-TR"}
        duts_te_train = {
            "name": "DUTS-TE",
            "im_dir": "cascade_psp/DUTS-TE",
            "gt_dir": "cascade_psp/DUTS-TE"}
        ecssd_train = {
            "name": "ECSSD",
            "im_dir": "cascade_psp/ecssd",
            "gt_dir": "cascade_psp/ecssd"}
        msra_train = {
            "name": "MSRA10K",
            "im_dir": "cascade_psp/MSRA_10K",
            "gt_dir": "cascade_psp/MSRA_10K"}

        # valid set: 1537 images
        dis_val = {
            "name": "DIS5K-VD",
            "im_dir": "DIS5K/DIS-VD/im",
            "gt_dir": "DIS5K/DIS-VD/gt"}
        thin_val = {
            "name": "ThinObject5k-TE",
            "im_dir": "thin_object_detection/ThinObject5K/images_test",
            "gt_dir": "thin_object_detection/ThinObject5K/masks_test"}
        coift_val = {
            "name": "COIFT",
            "im_dir": "thin_object_detection/COIFT/images",
            "gt_dir": "thin_object_detection/COIFT/masks"}
        hrsod_val = {
            "name": "HRSOD",
            "im_dir": "thin_object_detection/HRSOD/images",
            "gt_dir": "thin_object_detection/HRSOD/masks_max255"}

        if split == 'train':
            self.datasets = [dis_train, thin_train, fss_train, duts_train, 
                          duts_te_train, ecssd_train, msra_train]
        elif split == 'val':
            self.datasets = [dis_val, thin_val, coift_val, hrsod_val]
        else:
            raise ValueError(f'Undefined split: {split}')

        self.dataset_samples = []
        for idx, dataset in enumerate(self.datasets):
            image_path = self.dataset_path / dataset['im_dir']
            samples = [(x.stem, idx) for x in sorted(image_path.glob('*.jpg'))]
            self.dataset_samples.extend(samples)

        assert len(self.dataset_samples) > 0

    def get_sample(self, index) -> DSample:
        image_name, idx = self.dataset_samples[index]
        image_path = str(self.dataset_path / self.datasets[idx]['im_dir'] / f'{image_name}.jpg')
        mask_path = str(self.dataset_path / self.datasets[idx]['gt_dir'] / f'{image_name}.png')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = np.max(cv2.imread(mask_path).astype(np.int32), axis=2)
        instances_mask[instances_mask > 0] = 1

        return DSample(image, instances_mask, objects_ids=[1], sample_id=index)
    
# def unit_test():
#     dataset_train = HQSeg44kDataset('/playpen-raid2/qinliu/data/HQSeg44k', split='train')
#     num_samples_train = dataset_train.get_samples_number()

#     dataset_val = HQSeg44kDataset('/playpen-raid2/qinliu/data/HQSeg44k', split='val')
#     num_samples_val = dataset_val.get_samples_number()

#     assert num_samples_train == 44320 and num_samples_val == 1537


# if __name__ == '__main__':
#     unit_test()