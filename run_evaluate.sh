MODEL_PATH=./weights/simpleclick_models/sbd_vit_base.pth

python scripts/evaluate_model.py NoBRS \
--gpu=0 \
--checkpoint=./weights/focuscut_models/focuscut-resnet50.pth \
--eval-mode=cvpr \
--datasets=GrabCut