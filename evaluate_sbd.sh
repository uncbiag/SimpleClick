# MODEL_PATH=/playpen-raid2/qinliu/models/model_0908_2022/iter_mask/cocolvis_plainvit_large/004/checkpoints/054.pth
# MODEL_PATH=/playpen-raid2/qinliu/models/model_0908_2022/iter_mask/cocolvis_plainvit_base448/001/checkpoints/054.pth
# MODEL_PATH=/playpen-raid2/qinliu/models/model_0909_2022/iter_mask/cocolvis_plainvit_base448/000/checkpoints/229.pth
# MODEL_PATH=/playpen-raid2/qinliu/models/model_0909_2022/iter_mask/cocolvis_plainvit_base896/003/checkpoints/040.pth
# MODEL_PATH=/playpen-raid2/qinliu/models/model_0913_2022/iter_mask/cocolvis_plainvit_base448/001/checkpoints/last_checkpoint.pth
# MODEL_PATH=/playpen-raid2/qinliu/models/model_0915_2022/iter_mask/cocolvis_plainvit_base448/001/checkpoints/last_checkpoint.pth
# MODEL_PATH=/playpen-raid2/qinliu/models/model_0916_2022/iter_mask/sbd_plainvit_base448/008_lr_adjust/checkpoints/054.pth
# MODEL_PATH=/playpen-raid2/qinliu/models/model_0916_2022/iter_mask/sbd_plainvit_base448/016_lr_adjust/checkpoints/054.pth
# MODEL_PATH=/playpen-raid2/qinliu/models/model_0916_2022/iter_mask/sbd_plainvit_base448/015_upsample_x4/checkpoints/054.pth
# MODEL_PATH=/playpen-raid2/qinliu/models/model_0920_2022/iter_mask/sbd_plainvit_base448/002_sbd_baseline/checkpoints/054.pth
# MODEL_PATH=/playpen-raid2/qinliu/models/model_0920_2022/iter_mask/sbd_plainvit_base448/004_sbd_random_split/checkpoints/054.pth
# MODEL_PATH=/playpen-raid2/qinliu/models/model_0920_2022/iter_mask/cocolvis_plainvit_base448/000_cocolvis_baseline/checkpoints/054.pth
# MODEL_PATH=/playpen-raid2/qinliu/models/model_0921_2022/iter_mask/cocolvis_plainvit_base448/000_cocolvis_base_230epochs/checkpoints/150.pth
# MODEL_PATH=/playpen-raid2/qinliu/models/model_0922_2022/iter_mask/sbd_plainvit_base448/002_plainvit_base448_sbd_clicks_encoding_change/checkpoints/054.pth
# MODEL_PATH=/playpen-raid2/qinliu/models/model_0922_2022/iter_mask/sbd_plainvit_base448/005/checkpoints/last_checkpoint.pth
# MODEL_PATH=/playpen-raid2/qinliu/models/model_0923_2022/iter_mask/sbd_plainvit_base448/000_lr_1e-4/checkpoints/054.pth
# MODEL_PATH=/playpen-raid2/qinliu/models/model_0923_2022/iter_mask/cocolvis_plainvit_base448/000_base448_cocolvis_lr_5e-5/checkpoints/053.pth
# MODEL_PATH=/playpen-raid2/qinliu/models/model_0923_2022/iter_mask/cocolvis_vit_large448/002_cocolvis_vit_large448/checkpoints/054.pth
# MODEL_PATH=/playpen-raid2/qinliu/models/model_0923_2022/iter_mask/sbd_vit_large448/000_sbd_vit_large448/checkpoints/054.pth
# MODEL_PATH=/playpen-raid2/qinliu/models/model_0924_2022/iter_mask/sbd_vit_huge448/006_vit_huge448_sbd/checkpoints/054.pth
# MODEL_PATH=/playpen-raid2/qinliu/projects/iSegFormer/weights/cocolvis_vitl_epoch_54.pth
# MODEL_PATH=/playpen-raid2/qinliu/models/model_0924_2022/iter_mask/cocolvis_vit_huge448/000_cocolvis_vit_huge448/checkpoints/last_checkpoint.pth
# MODEL_PATH=/playpen-raid2/qinliu/models/model_0927_2022/iter_mask/cocolvis_vit_huge448/000_cocolvis_vith_short_training_ft_from_20epchs/checkpoints/009.pth

# MODEL_PATH=/playpen-raid2/qinliu/projects/iSegFormer/weights/sbd_vitb_epoch_54.pth
# python scripts/evaluate_model.py NoBRS \
# --gpu=0 \
# --checkpoint=${MODEL_PATH} \
# --eval-mode=cvpr \
# --iou-analysis

# MODEL_PATH=/playpen-raid2/qinliu/projects/iSegFormer/weights/sbd_vitl_epoch_54.pth
# python scripts/evaluate_model.py NoBRS \
# --gpu=0 \
# --checkpoint=${MODEL_PATH} \
# --eval-mode=cvpr \
# --iou-analysis

MODEL_PATH=/playpen-raid2/qinliu/projects/iSegFormer/weights/sbd_vith_epoch_54.pth
python scripts/evaluate_model.py NoBRS \
--gpu=0 \
--checkpoint=${MODEL_PATH} \
--eval-mode=cvpr \
--iou-analysis