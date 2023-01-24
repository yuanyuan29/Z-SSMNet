# Copyright 2023 Yuan Yuan, Biomedical Data Analysis and Visualisation (BDAV) Lab, The University of Sydney, Sydney, Australia

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

''' this code is used for zonal mask post-processing to remove noisy markers'''

import os
import SimpleITK as sitk
import numpy as np
import argparse

def mask_postprocessing(img: sitk.Image):
    # calculate connected components
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)
    out_mask = cc_filter.Execute(img)
    out_mask_array = sitk.GetArrayFromImage(out_mask)
    img_array = sitk.GetArrayFromImage(img)

    # if there is only one component, we are done
    if cc_filter.GetObjectCount() == 1:
        return img

    # if no, choose the largest component including the center point on y axis and set all others to zero
    elif cc_filter.GetObjectCount() > 1:
        cc_count = cc_filter.GetObjectCount()
        cc_count_list = list(np.arange(1, cc_count+1))
        # set the connected component only including same label 2 as 0
        for i in range(1, cc_filter.GetObjectCount() + 1):
            if img_array[out_mask_array == i].min() == 2:
                out_mask_array[out_mask_array == i] = 0
                cc_count -= 1
                cc_count_list.remove(i)
                      
        # find the largest component
        largest_component = np.argmax(
            [np.sum(out_mask_array == i) for i in cc_count_list]
        )
        # the label of the largest component
        largest_component_label = cc_count_list[largest_component]

        center_a = img_array.shape[2] // 2
        if img.GetSize()[0] >1000:
            prop = list(np.unique(out_mask_array[:, :, center_a]))
            prop.remove(0)
            # choose the prop nearest to the center of the image
            Prop_largest = np.argmax([np.sum(out_mask_array == i) for i in prop])
            Prop_largest_label = prop[Prop_largest]

            img_array[out_mask_array != Prop_largest_label] = 0
            img_out = sitk.GetImageFromArray(img_array)
            img_out.CopyInformation(img)
            return img_out
        
        # check here
        elif largest_component_label in out_mask_array[:, :, center_a]:
            # set all other components to zero
            img_array[out_mask_array != largest_component_label] = 0
            img_out = sitk.GetImageFromArray(img_array)
            img_out.CopyInformation(img)
            return img_out
    
        else:
            # only keep the largest component
            img_array[out_mask_array != largest_component_label] = 0
            img_out = sitk.GetImageFromArray(img_array)
            img_out.CopyInformation(img)
            return img_out
    else:
        print("no connection component found")
        return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--zonal_mask_dir', type=str, default='/workdir/results/nnUNet/3d_fullres/Task990_prostate_zonal_Seg/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/predictions/')
    parser.add_argument('--zonal_mask_post_dir', type=str, default='/workdir/results/nnUNet/3d_fullres/Task990_prostate_zonal_Seg/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/predictions_post')
    args = parser.parse_args()

    if not os.path.exists(args.zonal_mask_post_dir):
        os.makedirs(args.zonal_mask_post_dir)

    # post-processing for zonal mask
    for mask_name in os.listdir(args.zonal_mask_dir):
        if mask_name.endswith('.nii.gz'):
            mask_path = os.path.join(args.zonal_mask_dir, mask_name)
            mask = sitk.ReadImage(mask_path)
            mask = mask_postprocessing(mask)
            sitk.WriteImage(mask, os.path.join(args.zonal_mask_post_dir, mask_name))       
    print("well done!")





