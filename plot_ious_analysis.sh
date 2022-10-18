root_ours=/playpen-raid2/qinliu/models/model_0928_2022/evaluation_logs/others
root_ritm=/playpen-raid2/qinliu/models/model_0928_2022/evaluation_logs/others
root_foca=/playpen-raid2/qinliu/projects/ClickSEG/experiments/evaluation_logs/others

plot_path=/playpen-raid2/qinliu/projects/SimpleClick/results

# ${1} = GrabCut,Berkeley,SBD,DAVIS,PascalVOC,ssTEM,BraTS,OAIZIB
dataset=${1}

cl_vitb=${root_ours}/cocolvis_vitb_epoch_54/plots/${dataset}_cvpr_NoBRS_20.pickle
cl_vitl=${root_ours}/cocolvis_vitl_epoch_54/plots/${dataset}_cvpr_NoBRS_20.pickle
cl_vith=${root_ours}/cocolvis_vith_epoch_52/plots/${dataset}_cvpr_NoBRS_20.pickle

sbd_vitb=${root_ours}/sbd_vitb_epoch_54/plots/${dataset}_cvpr_NoBRS_20.pickle
sbd_vitl=${root_ours}/sbd_vitl_epoch_54/plots/${dataset}_cvpr_NoBRS_20.pickle
sbd_vith=${root_ours}/sbd_vith_epoch_54/plots/${dataset}_cvpr_NoBRS_20.pickle

cl_h32_ritm=${root_ritm}/cocolvis_h32_itermask/plots/${dataset}_cvpr_NoBRS_20.pickle
sbd_h18_ritm=${root_ritm}/sbd_h18_itermask/plots/${dataset}_cvpr_NoBRS_20.pickle

cl_segformer_b0_foca=${root_foca}/cocolvis_segformer_b0_s2/plots/${dataset}_cvpr_FocalClick_20.pickle
cl_segformer_b3_foca=${root_foca}/cocolvis_segformer_b3_s2/plots/${dataset}_cvpr_FocalClick_20.pickle

sbd_res34_cdnet=${root_foca}/sbd_cdnet_resnet34/plots/${dataset}_cvpr_CDNet_20.pickle
cl_res34_cdnet=${root_foca}/cocolvis_cdnet_resnet34/plots/${dataset}_cvpr_CDNet_20.pickle

python ./scripts/plot_ious_analysis.py \
  --files ${sbd_vitb} ${sbd_vitl} ${sbd_vith} ${cl_vitb} ${cl_vitl} ${cl_vith} ${sbd_h18_ritm} ${cl_h32_ritm} ${cl_segformer_b0_foca} ${cl_segformer_b3_foca} ${sbd_res34_cdnet} ${cl_res34_cdnet} \
  --datasets ${dataset} \
  --plots-path ${plot_path}