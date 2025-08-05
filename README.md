# [SimpleClick: Interactive Image Segmentation with Simple Vision Transformers](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_SimpleClick_Interactive_Image_Segmentation_with_Simple_Vision_Transformers_ICCV_2023_paper.html)

**University of North Carolina at Chapel Hill**

[Qin Liu](https://sites.google.com/cs.unc.edu/qinliu/home), [Zhenlin Xu](https://wildphoton.github.io/), [Gedas Bertasius](https://www.gedasbertasius.com/), [Marc Niethammer](https://biag.cs.unc.edu/)

ICCV 2023

<p align="center">
    <a href="https://paperswithcode.com/sota/interactive-segmentation-on-sbd?p=simpleclick-interactive-image-segmentation">
        <img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/simpleclick-interactive-image-segmentation/interactive-segmentation-on-sbd"/>
    </a>
</p>

<p align="center">
  <img src="./assets/simpleclick_framework.png" alt="drawing", width="650"/>
</p>


## Environment
Training and evaluation environment: Python3.8.8, PyTorch 1.11.0, Ubuntu 20.4, CUDA 11.0. Run the following command to install required packages.
```
pip3 install -r requirements.txt
```
You can build a container with the configured environment using our [Dockerfiles](https://github.com/uncbiag/SimpleClick/tree/v1.0/docker). 
Our Dockerfiles only support CUDA 11.0/11.4/11.6. If you use different CUDA drivers, you need to modify the base image in the Dockerfile (This is annoying that you need a matched image in Dockerfile for your CUDA driver, otherwise the gpu doesn't work in the container. Any better solutions?).
You also need to configue the paths to the datasets in [config.yml](https://github.com/uncbiag/SimpleClick/blob/v1.0/config.yml) before training or testing.

## Demo
<p align="center">
  <img src="./assets/demo_sheep.gif" alt="drawing", width="500"/>
</p>

An example script to run the demo. 
```
python3 demo.py --checkpoint=./weights/simpleclick_models/cocolvis_vit_huge.pth --gpu 0
```
Some test images can be found [here](https://github.com/uncbiag/SimpleClick/tree/v1.0/assets/test_imgs).

## Evaluation
Before evaluation, please download the datasets and models, and then configure the path in [config.yml](https://github.com/uncbiag/SimpleClick/blob/v1.0/config.yml).

Use the following code to evaluate the huge model.
```
python scripts/evaluate_model.py NoBRS \
--gpu=0 \
--checkpoint=./weights/simpleclick_models/cocolvis_vit_huge.pth \
--eval-mode=cvpr \
--datasets=GrabCut,Berkeley,DAVIS,PascalVOC,SBD,COCO_MVal,ssTEM,BraTS,OAIZIB
```

## Training
Before training, please download the [MAE](https://github.com/facebookresearch/mae) pretrained weights (click to download: [ViT-Base](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth), [ViT-Large](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth), [ViT-Huge](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth)) and configure the dowloaded path in [config.yml](https://github.com/uncbiag/SimpleClick/blob/main/config.yml).

Use the following code to train a huge model on C+L: 
```
python train.py models/iter_mask/plainvit_huge448_cocolvis_itermask.py \
--batch-size=32 \
--ngpus=4
```

## Model weights 
SimpleClick models: [Google Drive](https://drive.google.com/drive/folders/1zVhZefCjsTBxvyxnYMVnbkrNeRCH6y9Y?usp=sharing)

## Datasets

We train all our models on SBD and COCO+LVIS and evaluate them on GrabCut, Berkeley, DAVIS, SBD and PascalVOC. We also provide links to additional datasets: ADE20k and OpenImages, that are used in ablation study.

| Dataset   |                      Description             |           Download Link              |
|-----------|----------------------------------------------|:------------------------------------:|
|ADE20k     |  22k images with 434k instances (total)      |  [official site][ADE20k]             |
|OpenImages |  944k images with 2.6M instances (total)     |  [official site][OpenImages]         |
|MS COCO    |  118k images with 1.2M instances (train)     |  [official site][MSCOCO]             |
|LVIS v1.0  |  100k images with 1.2M instances (total)     |  [official site][LVIS]               |
|COCO+LVIS* |  99k images with 1.5M instances (train)      |  [original LVIS images][LVIS] + <br> [our combined annotations][COCOLVIS_annotation] |
|SBD        |  8498 images with 20172 instances for (train)<br>2857 images with 6671 instances for (test) |[official site][SBD]|
|Grab Cut   |  50 images with one object each (test)       |  [GrabCut.zip (11 MB)][GrabCut]      |
|Berkeley   |  96 images with 100 instances (test)         |  [Berkeley.zip (7 MB)][Berkeley]     |
|DAVIS      |  345 images with one object each (test)      |  [DAVIS.zip (43 MB)][DAVIS]          |
|Pascal VOC |  1449 images with 3417 instances (validation)|  [official site][PascalVOC]          |
|COCO_MVal  |  800 images with 800 instances (test)        |  [COCO_MVal.zip (127 MB)][COCO_MVal] |
|BraTS      |  369 cases (test)                            |  [BraTS20.zip (4.2 MB)][BraTS]       |
|OAI-ZIB    |  150 cases (test)                            |  [OAI-ZIB.zip (27 MB)][OAI-ZIB]      |

[ADE20k]: http://sceneparsing.csail.mit.edu/
[OpenImages]: https://storage.googleapis.com/openimages/web/download.html
[MSCOCO]: https://cocodataset.org/#download
[LVIS]: https://www.lvisdataset.org/dataset
[SBD]: http://home.bharathh.info/pubs/codes/SBD/download.html
[GrabCut]: https://drive.google.com/uc?export=download&id=1qKorUonIQcn3Z_IB6-en1K5q1K4T6pVK
[Berkeley]: https://drive.google.com/uc?export=download&id=1yo6PMKaMAu5jCCWf-Qf2boeG18b-m4vZ
[DAVIS]: https://drive.google.com/uc?export=download&id=1kyjN6EJSjwGnzSJxVjm3Pl2-XAjV7tac
[PascalVOC]: http://host.robots.ox.ac.uk/pascal/VOC/
[COCOLVIS_annotation]: https://drive.google.com/uc?export=download&id=17z9aZPlRv8vpU1AEz_M0WuZC6uBkqEWE
[COCO_MVal]: https://drive.google.com/uc?export=download&id=1_TgkjAmmpJLMIYSBRp89gaPNrFp_XxW5
[BraTS]: https://drive.google.com/uc?export=download&id=1uaveX_nziTLaJOj-Gl3csvIaa6Q__EhP
[OAI-ZIB]: https://drive.google.com/uc?export=download&id=11N6pJL5HowogUriCXVFbK3GacPL2X2Xx

Don't forget to change the paths to the datasets in [config.yml](config.yml) after downloading and unpacking.

(*) To prepare COCO+LVIS, you need to download original LVIS v1.0, then download and unpack our 
pre-processed annotations that are obtained by combining COCO and LVIS dataset into the folder with LVIS v1.0.


## Notes
[03/11/2023] Add an xTiny model.

[10/25/2022] Add docker files.

[10/02/2022] Release the main models. This repository is still under active development.

## License
The code is released under the MIT License. It is a short, permissive software license. Basically, you can do whatever you want as long as you include the original copyright and license notice in any copy of the software/source. 

## Citation
```bibtex
@InProceedings{Liu_2023_ICCV,
    author    = {Liu, Qin and Xu, Zhenlin and Bertasius, Gedas and Niethammer, Marc},
    title     = {SimpleClick: Interactive Image Segmentation with Simple Vision Transformers},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {22290-22300}
}
```
## Acknowledgement
Our project is developed based on [RITM](https://github.com/saic-vul/ritm_interactive_segmentation). Thanks for the nice demo GUI :)
