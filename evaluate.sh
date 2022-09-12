# MODEL_PATH=/playpen-raid2/qinliu/models/model_0908_2022/iter_mask/cocolvis_plainvit_large/004/checkpoints/054.pth
#MODEL_PATH=/playpen-raid2/qinliu/models/model_0908_2022/iter_mask/cocolvis_plainvit_base448/001/checkpoints/054.pth
MODEL_PATH=/playpen-raid2/qinliu/models/model_0909_2022/iter_mask/cocolvis_plainvit_base448/000/checkpoints/229.pth

python scripts/evaluate_model.py NoBRS \
--gpu=1 \
--checkpoint=${MODEL_PATH} \
--eval-mode=fixed448
# --datasets=OAIZIB,COCO_MVal
# --vis-preds

