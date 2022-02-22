# # Train PASCAL HRNet32
# python train.py models/iter_mask/hrnet32_pascal_itermask_3p.py \
# --batch-size=32 \
# --gpus=3 \
# --exp-name hrnet32_pascal


# # Train SBD HRNet18
# python train.py models/iter_mask/hrnet18_sbd_itermask_3p.py \
# --batch-size=32 \
# --gpus=1 \
# --exp-name hrnet18_sota_finetune \
# --weights ./weights/sbd_h18_itermask.pth


# # Train SBD HRNet32
# python train.py models/iter_mask/hrnet32_sbd_itermask_3p.py \
# --batch-size=32 \
# --gpus=3 \
# --exp-name hrnet32_sbd


# # Train COCO_LVIS HRNet18
# python train.py models/iter_mask/hrnet18_cocolvis_itermask_3p.py \
# --batch-size=32 \
# --gpus=2 \
# --weights ./weights/sbd_h18_itermask_sota.pth


# # Train COCO_LVIS HRNet32
# python train.py models/iter_mask/hrnet32_cocolvis_itermask_3p.py \
# --batch-size=32 \
# --gpus=3 \
# --exp-name cocolvis_hrnet32
# --weights ./weights/coco_lvis_h32_itermask.pth


# # Train SegFormer
# python train.py models/iter_mask/segformerb2_cocolvis_itermask_v4.py --batch-size=8 --gpus=2 --exp-name cocolvis


# # Train HRFormer
# python train.py models/iter_mask/hrformer_base_cocolvis_itermask_v3.py --batch-size=8 --gpus=1 --exp-name cocolvis_aux

# Train SwinFormer
#python train.py models/iter_mask/swinformer_large_cocolvis_itermask.py --batch-size=22 --gpus=0 --exp-name cocolvis_swin_large
#python train.py models/iter_mask/swinformer_base_cocolvis_itermask.py --batch-size=32 --gpus=1 --exp-name cocolvis_swin_base

# Fine tune on OAI-ZIB dataset
python train.py models/iter_mask/swinformer_large_oaizib_itermask.py \
--batch-size=22 \
--gpu=1 \
--exp-name oaizib_swin_large
#--weights /playpen-raid2/qinliu/projects/iSegFormer/weights/oai_pretrain_swin_large_epoch_54.pth
