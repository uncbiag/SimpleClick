MODEL_PATH=/playpen-raid2/qinliu/models/model_0908_2022/iter_mask/cocolvis_plainvit_large/004/checkpoints/054.pth

python scripts/evaluate_model.py NoBRS \
--gpu=1 \
--checkpoint=${MODEL_PATH} \
--eval-mode=fixed224 
# --vis-preds
