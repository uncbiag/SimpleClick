from isegm.data.datasets import PASCAL
from isegm.data.points_sampler import MultiClassSampler
import argparse
import os
from pathlib import Path

from isegm.data.points_sampler import MultiClassSampler
from isegm.engine.Multi_trainer import Multi_trainer
from isegm.inference.clicker import Click
from isegm.model.is_plainvit_model import MultiOutVitModel
from isegm.model.metrics import AdaptiveMIoU
from isegm.utils.exp import init_experiment
from isegm.utils.exp_imports.default import *
from isegm.model.modeling.transformer_helper.cross_entropy_loss import CrossEntropyLoss
from train import load_module

points_sampler = MultiClassSampler(2, prob_gamma=0.80,
                                   merge_objects_prob=0.15,
                                   max_num_merged_objects=2)
trainset = PASCAL(
    "/home/gyt/gyt/dataset/data/pascal_person_part",
    split='train',
    min_object_area=1000,
    keep_background_prob=0.05,
    points_sampler=points_sampler,
    epoch_len=30000,
    # stuff_prob=0.30
)

valset = PASCAL(
    "/home/gyt/gyt/dataset/data/pascal_person_part",
    split='val',
    min_object_area=1000,
    points_sampler=points_sampler,
    epoch_len=2000
)

for batch_data in trainset:
    print(batch_data["points"].shape)