
# MODEL_PATH=/playpen-raid2/qinliu/projects/iSegFormer/weights/cocolvis_vitb_epoch_54.pth
MODEL_PATH=/playpen-raid2/qinliu/models/model_0924_2022/iter_mask/cocolvis_vit_huge448/000_cocolvis_vit_huge448/checkpoints/052.pth

python scripts/evaluate_model.py NoBRS \
--gpu=0 \
--checkpoint=${MODEL_PATH} \
--eval-mode=cvpr \
--iou-analysis

# MODEL_PATH=/playpen-raid2/qinliu/models/model_0924_2022/iter_mask/cocolvis_vit_huge448/000_cocolvis_vit_huge448/checkpoints/053.pth

# python scripts/evaluate_model.py NoBRS \
# --gpu=0 \
# --checkpoint=${MODEL_PATH} \
# --eval-mode=cvpr \
# --iou-analysis
