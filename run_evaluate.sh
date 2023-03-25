# MODEL_PATH=/playpen-raid2/qinliu/projects/SimpleClick/weights/simpleclick_models/cocolvis_vitb_epoch_54.pth
# MODEL_PATH=/playpen-raid2/qinliu/models/model_0125_2023/iter_mask/sbd_plainvit_base448/006_lr1e-4_ft/checkpoints
# MODEL_PATH=/playpen-raid2/qinliu/models/model_0125_2023/iter_mask/sbd_plainvit_base448/010/checkpoints
# MODEL_PATH=/playpen-raid2/qinliu/models/model_0323_2023/referring/ade20k_plainvit_base448/000/checkpoints
MODEL_PATH=/playpen-raid2/qinliu/models/model_0324_2023/iter_mask/sbd_plainvit_base448/007/checkpoints/054.pth

python scripts/evaluate_model.py NoBRS \
--gpu=0 \
--checkpoint=${MODEL_PATH} \
--eval-mode=cvpr \
--datasets=GrabCut,SBD