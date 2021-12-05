#model_folder=/playpen-raid/qinliu/models/model_1025_2021/iter_mask/cocolvis_hrnet32/003_hrnet32_sota_finetune/checkpoints
model_folder=/playpen-raid/qinliu/models/model_1025_2021/iter_mask/sbd_hrnet18/000_hrnet18_sota_finetune/checkpoints

for file in 040 045 050 055 060 065 070
do
    echo 'Testing epoch: ' ${file}
    python scripts/evaluate_model.py NoBRS \
    --gpus 1 \
    --checkpoint=${model_folder}/${file}.pth  \
    --dataset=GrabCut,Berkeley,DAVIS,PascalVOC,SBD

done
