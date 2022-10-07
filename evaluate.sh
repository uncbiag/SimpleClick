MODEL_PATH=./weights/simpleclick_models/cocolvis_vitb_epoch_54.pth
# MODEL_PATH=./weights/ritm_models/sbd_h18_itermask.pth

python scripts/evaluate_model.py NoBRS \
--gpu=0 \
--checkpoint=${MODEL_PATH} \
--eval-mode=cvpr \
--datasets=GrabCut
#  \
# --eval-ritm
# --vis-preds \
# --iou-analysis
# --n-clicks 100
