import numpy as np
import os
import SimpleITK as sitk
import torch
from model.propagation.prop_net import PropagationNetwork
from model.fusion_net import FusionNet
from interact.interactive_utils import load_volume, images_to_torch
from inference_core import InferenceCore

model_folder = '/playpen-raid2/qinliu/projects/iSegFormer/maskprop/MiVOS/saves'
prop_model_path = os.path.join(model_folder, 'stcn.pth')
fusion_model_path = os.path.join(model_folder, 'fusion_stcn.pth')

# Load our checkpoint
prop_saved = torch.load(prop_model_path)
prop_model = PropagationNetwork().cuda().eval()
prop_model.load_state_dict(prop_saved)

fusion_saved = torch.load(fusion_model_path)
fusion_model = FusionNet().cuda().eval()
fusion_model.load_state_dict(fusion_saved)

volume_path = '/playpen-raid2/qinliu/projects/iSegFormer/maskprop/MiVOS/example/9092628_image.nii.gz'
images = load_volume(volume_path, normalize=True)

label_path = '/playpen-raid2/qinliu/projects/iSegFormer/maskprop/MiVOS/example/9092628_label.nii.gz'
label = load_volume(label_path, normalize=False)

num_objects = 2
mem_freq = 5
mem_profile = 0

processor = InferenceCore(
    prop_model, 
    fusion_model, 
    images_to_torch(images, device='cpu'), 
    num_objects, 
    mem_freq=mem_freq, 
    mem_profile=mem_profile
)


def get_selected_mask(label, frame_idx):
    label_frame = label[label_frame_idx][:,:,0]

    label_frame_fc = np.zeros_like(label_frame)
    label_frame_fc[label_frame == 2] = 1

    label_frame_tc = np.zeros_like(label_frame)
    label_frame_tc[label_frame == 4] = 1

    label_frame_bg = np.ones_like(label_frame)
    label_frame_bg[label_frame_fc] = 0
    label_frame_bg[label_frame_tc] = 0

    label_frame = np.stack([label_frame_bg, label_frame_fc, label_frame_tc], axis=0)
    return label_frame


mask_save_folder = '/playpen-raid2/qinliu/projects/iSegFormer/maskprop/MiVOS/test_results'

label_frame_idx = 40
label_frame = get_selected_mask(label, label_frame_idx)
label_frame = torch.from_numpy(label_frame)
label_frame = torch.unsqueeze(label_frame, dim=1)
current_mask = processor.interact(label_frame, label_frame_idx)

# save the propagated mask with only one labeled frame
current_mask = sitk.GetImageFromArray(current_mask)
current_mask.CopyInformation(sitk.ReadImage(label_path))
current_mask_save_path = os.path.join(mask_save_folder, '9092628_seg_1frame.nii.gz')
sitk.WriteImage(current_mask, current_mask_save_path)

label_frame_idx = 120
label_frame = get_selected_mask(label, label_frame_idx)
label_frame = torch.from_numpy(label_frame)
label_frame = torch.unsqueeze(label_frame, dim=1)
current_mask = processor.interact(label_frame, label_frame_idx)

# save the propagated mask with two labeled frames
current_mask = sitk.GetImageFromArray(current_mask)
current_mask.CopyInformation(sitk.ReadImage(label_path))
current_mask_save_path = os.path.join(mask_save_folder, '9092628_seg_2frames.nii.gz')
sitk.WriteImage(current_mask, current_mask_save_path)

label_frame_idx = 80
label_frame = get_selected_mask(label, label_frame_idx)
label_frame = torch.from_numpy(label_frame)
label_frame = torch.unsqueeze(label_frame, dim=1)
current_mask = processor.interact(label_frame, label_frame_idx)

# save the propagated mask with two labeled frames
current_mask = sitk.GetImageFromArray(current_mask)
current_mask.CopyInformation(sitk.ReadImage(label_path))
current_mask_save_path = os.path.join(mask_save_folder, '9092628_seg_3frames.nii.gz')
sitk.WriteImage(current_mask, current_mask_save_path)
