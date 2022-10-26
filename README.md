## [SimpleClick: Interactive Image Segmentation with Simple Vision Transformers](https://arxiv.org/abs/2210.11006)
<p align="center">
    <a href="https://paperswithcode.com/sota/interactive-segmentation-on-sbd?p=simpleclick-interactive-image-segmentation">
        <img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/simpleclick-interactive-image-segmentation/interactive-segmentation-on-sbd"/>
    </a>
</p>

<p align="center">
  <img src="./assets/img/simpleclick_framework.png" alt="drawing", width="650"/>
</p>

<p align="center">
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/License-MIT-yellow.svg"/>
    </a>
    <a href="https://arxiv.org/pdf/2210.11006.pdf">
        <img src="https://img.shields.io/badge/arXiv-2102.06583-b31b1b"/>
    </a>    
</p>

## Environment
The models are trained and evaluated using Python 3.8.8 and PyTorch 1.11.0.
But we find the evaluation results can be reproduced with the docker using Python 3.6.9 and PyTorch 1.9.0.
```
pip3 install -r requirements.txt
```

## Demo
<p align="center">
  <img src="./assets/demo_sheep.gif" alt="drawing", width="500"/>
</p>

```
python3 demo.py 
--checkpoint=./weights/simpleclick_models/cocolvis_vit_huge.pth
--gpu 0
```


## Download 
Pre-trained SimpleClick models: [Google Drive](https://drive.google.com/drive/folders/1qpK0gtAPkVMF7VC42UA9XF4xMWr5KJmL?usp=sharing)

BraTS dataset (369 cases): [Google Drive](https://drive.google.com/drive/folders/1B6y1nNBnWU09EhxvjaTdp1XGjc1T6wUk?usp=sharing) 

OAI-ZIB dataset (150 cases): [Google Drive](https://drive.google.com/drive/folders/1B6y1nNBnWU09EhxvjaTdp1XGjc1T6wUk?usp=sharing)

Other datasets: [RITM Github](https://github.com/saic-vul/ritm_interactive_segmentation)

## Notes
[10/02/2022] Release the main models. This repository is still under active development.

## License
The code is released under the MIT License. It is a short, permissive software license. Basically, you can do whatever you want as long as you include the original copyright and license notice in any copy of the software/source. 

## Citation
```bibtex
@article{liu2022simpleclick,
  title={SimpleClick: Interactive Image Segmentation with Simple Vision Transformers},
  author={Liu, Qin and Xu, Zhenlin and Bertasius, Gedas and Niethammer, Marc},
  journal={arXiv preprint arXiv:2210.11006},
  year={2022}
}
```
## Acknowledgement
Our project is developed based on [RITM](https://github.com/saic-vul/ritm_interactive_segmentation). Thanks for the nice demo GUI :)
