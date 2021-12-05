#model_folder=/playpen-raid/qinliu/models/model_1025_2021/iter_mask/sbd_hrnet18/000_hrnet18_sota_finetune/checkpoints/124.pth
#model_folder=/playpen-raid/qinliu/models/model_1025_2021/iter_mask/cocolvis_hrnet32/003_hrnet32_sota_finetune/checkpoints/090.pth
model_folder=/playpen-raid/qinliu/projects/ritm_interactive_segmentation/weights/coco_lvis_h32_itermask.pth
#model_folder=/playpen-raid/qinliu/models/model_1113_2021/iter_mask/cocolvis_hrnet18/003/checkpoints/035.pth
#model_folder=/playpen-raid/qinliu/models/model_1113_2021/iter_mask/sbd_hrnet32/001_hrnet32_sbd/checkpoints/last_checkpoint.pth
python scripts/evaluate_model.py NoBRS \
--gpus 0 \
--checkpoint=${model_folder} \
--dataset=BraTS --print-ious --n-clicks 5
# --dataset=GrabCut,Berkeley,DAVIS,PascalVOC,SBD
