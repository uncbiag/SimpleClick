model1=/playpen-raid2/qinliu/models/model_0927_2022/evaluation_logs/others/sbd_vitb_epoch_54
model2=/playpen-raid2/qinliu/models/model_0927_2022/evaluation_logs/others/sbd_vitl_epoch_54
model3=/playpen-raid2/qinliu/models/model_0927_2022/evaluation_logs/others/sbd_vith_epoch_54

python ./scripts/plot_ious_analysis.py \
  --model-dirs ${model1} ${model2} ${model3} \
  --mode NoBRS
