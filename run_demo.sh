model_folder=./weights
model_name=cocolvis_vith_epoch_52.pth

python3 demo.py \
--checkpoint=${model_folder}/${model_name} \
--gpu 0