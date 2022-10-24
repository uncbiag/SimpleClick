# Run demo for SimpleClick models
model_folder=./weights/simpleclick_models
model_name=cocolvis_vitl_epoch_54.pth

python3 demo.py \
--checkpoint=${model_folder}/${model_name} \
--gpu 0


# # Run demo for RITM models
# model_folder=./weights/ritm_models
# model_name=coco_lvis_h32_itermask.pth

# python3 demo.py \
# --checkpoint=${model_folder}/${model_name} \
# --gpu 0 \
# --eval-ritm 