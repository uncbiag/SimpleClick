
# MODEL_PATH=/playpen-raid2/qinliu/projects/iSegFormer/weights/cocolvis_vitb_epoch_54.pth
# MODEL_PATH=/playpen-raid2/qinliu/models/model_0924_2022/iter_mask/cocolvis_vit_huge448/000_cocolvis_vit_huge448/checkpoints/052.pth
# MODEL_PATH=/playpen-raid/qinliu/projects/ritm_interactive_segmentation/weights/sbd_h18_itermask.pth
MODEL_PATH=./weights/cocolvis_vitl_epoch_54.pth

python scripts/evaluate_model.py NoBRS \
--gpu=0 \
--checkpoint=${MODEL_PATH} \
--eval-mode=cvpr \
--datasets=GrabCut \
--vis-preds \
--iou-analysis
# --n-clicks 100
