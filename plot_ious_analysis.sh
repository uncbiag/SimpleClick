root_ours=/playpen-raid2/qinliu/models/model_0928_2022/evaluation_logs/others
root_sota=/playpen-raid2/qinliu/models/model_0929_2022/evaluation_logs/others

# ${1} = GrabCut,Berkeley,SBD,DAVIS,PascalVOC,ssTEM,BraTS,OAIZIB
dataset=${1}

cl_vitb=${root_ours}/cocolvis_vitb_epoch_54/plots/${dataset}_cvpr_NoBRS_20.pickle
cl_vitl=${root_ours}/cocolvis_vitl_epoch_54/plots/${dataset}_cvpr_NoBRS_20.pickle
cl_vith=${root_ours}/052/plots/${dataset}_cvpr_NoBRS_20.pickle

sbd_vitb=${root_ours}/sbd_vitb_epoch_54/plots/${dataset}_cvpr_NoBRS_20.pickle
sbd_vitl=${root_ours}/sbd_vitl_epoch_54/plots/${dataset}_cvpr_NoBRS_20.pickle
sbd_vith=${root_ours}/sbd_vith_epoch_54/plots/${dataset}_cvpr_NoBRS_20.pickle

cl_h32_ritm=${root_sota}/coco_lvis_h32_itermask/plots/${dataset}_cvpr_NoBRS_20.pickle
sbd_h18_ritm=${root_sota}/sbd_h18_itermask/plots/${dataset}_cvpr_NoBRS_20.pickle

python ./scripts/plot_ious_analysis.py \
  --files ${cl_vitb} ${cl_vitl} ${cl_vith} ${sbd_vitb} ${sbd_vitl} ${sbd_vith} ${cl_h32_ritm} ${sbd_h18_ritm} \
  --mode NoBRS \
  --datasets ${dataset}
