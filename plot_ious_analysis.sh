root=/playpen-raid2/qinliu/models/model_0928_2022/evaluation_logs/others
sbd_vitb_grabcut=${root}/sbd_vitb_epoch_54/plots/GrabCut_cvpr_NoBRS_20.pickle
sbd_vitl_grabcut=${root}/sbd_vitl_epoch_54/plots/GrabCut_cvpr_NoBRS_20.pickle
sbd_vith_grabcut=${root}/sbd_vith_epoch_54/plots/GrabCut_cvpr_NoBRS_20.pickle
# cocolvis_vitb=${root}/cocolvis_vitb_epoch_54
# cocolvis_vitl=${root}/cocolvis_vitl_epoch_54
# # cocolvis_vith=${root}/cocolvis_vith_epoch_54
# cocolvis_vith=${root}/052

python ./scripts/plot_ious_analysis.py \
  --files ${sbd_vitb_grabcut} ${sbd_vitl_grabcut} ${sbd_vith_grabcut} \
  --mode NoBRS \
  --datasets GrabCut
