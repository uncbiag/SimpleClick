model_folder=/playpen-raid/qinliu/projects/iSegFormer/weights
model_name=cocolvis_hr32_epoch_54.pth

python3 demo.py \
--checkpoint=${model_folder}/${model_name} \
--gpu 1
