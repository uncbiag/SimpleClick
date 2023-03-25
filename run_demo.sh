MODEL_PATH=./weights/simpleclick_models/cocolvis_vitl_epoch_54.pth
#MODEL_PATH=./weights/simpleclick_models/sbd_vitxt.pth
python3 demo.py \
--checkpoint=${MODEL_PATH} \
--gpu 3
