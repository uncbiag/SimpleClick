import sys

sys.path.insert(0, '/playpen-raid2/qinliu/projects/iSegFormer')

from isegm.data.datasets.grabcut import GrabCutDataset
from isegm.data.compose import ComposeDataset, ProportionalComposeDataset
from isegm.data.datasets.berkeley import BerkeleyDataset
from isegm.data.datasets.coco import CocoDataset
from isegm.data.datasets.davis import DavisDataset
from isegm.data.datasets.grabcut import GrabCutDataset
from isegm.data.datasets.coco_lvis import CocoLvisDataset
from isegm.data.datasets.lvis import LvisDataset
from isegm.data.datasets.lvis_v1 import Lvis_v1_Dataset
from isegm.data.datasets.openimages import OpenImagesDataset
from isegm.data.datasets.sbd import SBDDataset, SBDEvaluationDataset
from isegm.data.datasets.images_dir import ImagesDirDataset
from isegm.data.datasets.ade20k import ADE20kDataset
from isegm.data.datasets.pascalvoc import PascalVocDataset
from isegm.data.datasets.brats import BraTSDataset
from isegm.data.datasets.ssTEM import ssTEMDataset
from isegm.data.datasets.oai_zib import OAIZIBDataset
from isegm.data.datasets.oai import OAIDataset

# Evaluation datasets
GRABCUT_PATH="/playpen-raid2/qinliu/data/GrabCut"
BERKELEY_PATH="/playpen-raid/qinliu/data/Berkeley"
DAVIS_PATH="/playpen-raid/qinliu/data/DAVIS"
COCO_MVAL_PATH="/playpen-raid/qinliu/data/COCO_MVal"

BraTS_PATH="/playpen-raid/qinliu/data/BraTS20"
ssTEM_PATH="/playpen-raid/qinliu/data/ssTEM"

OAIZIB_PATH="/playpen-raid2/qinliu/data/OAI-ZIB"
OAI_PATH="/playpen-raid2/qinliu/data/OAI"
SBD_PATH="/playpen-raid/qinliu/data/SBD/dataset"
PASCALVOC_PATH="/playpen-raid/qinliu/data/PascalVOC"


def get_dataset(dataset_name):
    if dataset_name == 'GrabCut':
        dataset = GrabCutDataset(GRABCUT_PATH)
    elif dataset_name == 'Berkeley':
        dataset = BerkeleyDataset(BERKELEY_PATH)
    elif dataset_name == 'DAVIS':
        dataset = DavisDataset(DAVIS_PATH)
    elif dataset_name == 'SBD':
        dataset = SBDEvaluationDataset(SBD_PATH)
    elif dataset_name == 'SBD_Train':
        dataset = SBDEvaluationDataset(SBD_PATH, split='train')
    elif dataset_name == 'PascalVOC':
        dataset = PascalVocDataset(PASCALVOC_PATH, split='val')
    elif dataset_name == 'COCO_MVal':
        dataset = DavisDataset(COCO_MVAL_PATH)
    elif dataset_name == 'BraTS':
        dataset = BraTSDataset(BraTS_PATH)
    elif dataset_name == 'ssTEM':
        dataset = ssTEMDataset(ssTEM_PATH)
    elif dataset_name == 'OAIZIB':
        dataset = OAIZIBDataset(OAIZIB_PATH)
    else:
        dataset = None

    return dataset


GrabCut = get_dataset('GrabCut')
Berkeley = get_dataset('Berkeley')
DAVIS = get_dataset('DAVIS')
SBD = get_dataset('SBD')
PascalVOC = get_dataset('PascalVOC')
COCO_MVal = get_dataset('COCO_MVal')
BraTS = get_dataset('BraTS')
ssTEM = get_dataset('ssTEM')
OAIZIB = get_dataset('OAIZIB')

print('Length of each evaluation dataset.')
# print('GrabCut: ', len(GrabCut))
# print('Berkeley: ', len(Berkeley))
# print('DAVIS: ', len(DAVIS))
# print('SBD: ', len(SBD))
# print('PascalVOC: ', len(PascalVOC))
# print('COCO_MVal: ', len(COCO_MVal))
# print('BraTS: ', len(BraTS))
# print('ssTEM: ', len(ssTEM))
# print('OAIZIB: ', len(OAIZIB))

dataset_names = ['GrabCut']
# dataset_names = ['GrabCut', 'Berkeley', 'DAVIS', 'SBD', 'PascalVOC', 'COCO_MVal', 'BraTS', 'ssTEM', 'OAIZIB']
xs, ys, labels = [], [], []
for dataset_name in dataset_names:
    dataset = get_dataset(dataset_name)
    print(dataset_name, len(dataset))
    for i in range(len(dataset)):
        sample = dataset.get_sample(i)
        print(sample.image.shape)
        x, y, _ = sample.image.shape
        xs.append(x)
        ys.append(y)
        labels.append(dataset_name)

import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(dict(x=xs, y=ys, label=labels))
groups = df.groupby('label')

fig, ax = plt.subplots()
# ax.margins(0.5)
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', label=name)
ax.legend()
ax.grid()

plt.show()