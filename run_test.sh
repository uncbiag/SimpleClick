#model_folder=/playpen-raid/qinliu/models/model_1025_2021/iter_mask/sbd_hrnet18/000_hrnet18_sota_finetune/checkpoints/124.pth
#model_folder=/playpen-raid/qinliu/models/model_1025_2021/iter_mask/cocolvis_hrnet32/003_hrnet32_sota_finetune/checkpoints/090.pth
#model_folder=/playpen-raid/qinliu/projects/ritm_interactive_segmentation/weights/coco_lvis_h32_itermask.pth
#model_folder=/playpen-raid/qinliu/models/model_1113_2021/iter_mask/cocolvis_hrnet18/003/checkpoints/035.pth
#model_folder=/playpen-raid/qinliu/models/model_1113_2021/iter_mask/sbd_hrnet32/001_hrnet32_sbd/checkpoints/last_checkpoint.pth
#model_folder=/playpen-raid/qinliu/models/model_0207_2022/iter_mask/cocolvis_swinformer_base/000_cocolvis_swin_base/checkpoints/054.pth
# python scripts/evaluate_model.py NoBRS \
# --gpus 0 \
# --checkpoint=${model_folder} \
# --dataset=GrabCut,Berkeley,DAVIS,BraTS,ssTEM,PascalVOC,SBD

# --dataset=GrabCut --vis-preds

# --dataset=GrabCut,Berkeley,DAVIS,PascalVOC,SBD

# --dataset=BraTS --print-ious --n-clicks 5

#model_folder=/playpen-raid/qinliu/models/model_0207_2022/iter_mask/cocolvis_hrnet32/002_cocolvis_hrnet32/checkpoints/054.pth
#model_folder=/playpen-raid/qinliu/models/model_0218_2022/iter_mask/oaizib_hrnet32/004_oaizib_hrnet32/checkpoints/030.pth
#model_folder=/playpen-raid/qinliu/models/model_0218_2022/iter_mask/oaizib_swinformer_large/003_oaizib_swin_large/checkpoints/030.pth
#model_folder=/playpen-raid/qinliu/models/model_0221_2022/iter_mask/oaizib_hrnet32/000_oaizib_hrnet32_finetune/checkpoints/054.pth
#model_folder=/playpen-raid/qinliu/models/model_0221_2022/iter_mask/oaizib_swinformer_base/000_oaizib_swin_base_finetune/checkpoints/030.pth
#model_folder=/playpen-raid2/qinliu/projects/iSegFormer/weights/oai_pretrain_swin_large_epoch_54.pth
#model_folder=/playpen-raid/qinliu/models/model_0221_2022/iter_mask/oaizib_swinformer_large/001_oaizib_swin_large_finetune/checkpoints/054.pth
#model_folder=/playpen-raid/qinliu/models/model_0221_2022/iter_mask/oaizib_swinformer_large/004_oaizib_swin_large/checkpoints/054.pth
#model_folder=/playpen-raid/qinliu/models/model_0221_2022/iter_mask/oaizib_hrnet32/003_oaizib_hrnet32/checkpoints/054.pth
#model_folder=/playpen-raid/qinliu/models/model_0221_2022/iter_mask/oaizib_swinformer_base/001_oaizib_swin_base/checkpoints/054.pth
#model_folder=/playpen-raid/qinliu/models/model_0228_2022/iter_mask/oaizib_hrnet32/000_oaizib_hrnet32_finetune/checkpoints/054.pth
#model_folder=/playpen-raid/qinliu/models/model_0228_2022/iter_mask/oaizib_swinformer_large/000_no_pretrain_oaizib_ft_swin_large/checkpoints/054.pth
model_folder=/playpen-raid2/qinliu/projects/iSegFormer/weights/imagenet21k_pretrain_cocolvis_finetune_segformerb5_epoch_54.pth


python scripts/evaluate_model.py NoBRS \
--gpu 0 \
--checkpoint=${model_folder} \
--dataset=OAIZIB
