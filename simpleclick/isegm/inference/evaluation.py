import numpy as np
from time import time
from tqdm import tqdm
import torch
from typing import Any, Dict, List, Optional, Tuple

from isegm.inference.utils import get_iou
from isegm.inference.clicker import Clicker
from isegm.inference.predictor import BasePredictor


def evaluate_dataset(
    dataset, 
    predictor: BasePredictor, 
    **kwargs
):
    all_ious = []

    start_time = time()
    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)

        for object_id in sample.objects_ids:
            sample_ious = evaluate_sample(
                sample.image, 
                sample.gt_mask(object_id), 
                predictor, 
                sample_id=index, 
                **kwargs,
            )
            all_ious.append(sample_ious)
    end_time = time()
    elapsed_time = end_time - start_time

    return all_ious, elapsed_time

# from torchvision.transforms.functional import resize, to_pil_image  # type: ignore

# def evaluate_dataset(
#     dataset, 
#     predictor: BasePredictor, 
#     **kwargs
# ):
#     sample = dataset.get_sample(0)
#     object_id = sample.objects_ids[0]

#     image = sample.image
#     gt_mask = sample.gt_mask(object_id)

#     # resize image and gt_mask to 1024 x 1024
#     image = np.array(resize(to_pil_image(image), (1024, 1024)))
#     gt_mask = np.array(resize(to_pil_image(gt_mask), (1024, 1024)))
#     pred_mask = np.zeros_like(gt_mask)

#     clicker = Clicker(gt_mask=gt_mask)
#     clicker.make_next_click(pred_mask)


#     num_trails = 1
#     elapse_time = 0
#     for _ in range(num_trails):
#         start_time = time()
#         # SAT
#         predictor.set_image(image)
#         for _ in range(256):
#             print(_)
#             _ = predictor.predict(clicker)

#         end_time = time()
        
#         elapse_time += end_time - start_time
#     print(elapse_time / num_trails)

#     return [], 0


def evaluate_sample(
    image: np.ndarray, 
    gt_mask: np.ndarray, 
    predictor: BasePredictor, 
    max_iou_thr: float,
    pred_thr: float=0.49, 
    min_clicks: int=1, 
    max_clicks: int=20,
    sample_id=None, 
    callback=None,
) -> np.ndarray:
    """
    Evaluate a sample. 
    image: H x W x 3, uint8
    gt_mask: H x W, int32
    """
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask)
    ious_list = []

    with torch.no_grad():
        predictor.set_image(image)

        for click_indx in range(max_clicks):
            clicker.make_next_click(pred_mask)
            pred_probs = predictor.predict(clicker)
            pred_mask = pred_probs > pred_thr

            if callback is not None:
                callback(image, gt_mask, pred_probs, sample_id, 
                         click_indx, clicker.clicks_list)

            iou = get_iou(gt_mask, pred_mask)
            ious_list.append(iou)

            if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                break

        return np.array(ious_list, dtype=np.float32)
