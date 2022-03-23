model_folder=/playpen-raid2/qinliu/projects/iSegFormer/weights
#model_name=cocolvis_hr32_epoch_54.pth
#model_name=oaizib_swin_base_epoch_54.pth
model_name=imagenet21k_pretrain_oaizib_finetune_swin_base_epoch_54.pth


python3 demo.py \
--checkpoint=${model_folder}/${model_name} \
--gpu 1
