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

import argparse
import os
from pathlib import Path
import pickle
import stat
from typing import Union

import numpy as np
import SimpleITK as sitk


def prepare_zonal_mask_npz(
    data_path: Union[Path, str],
    save_path: Union[Path, str],
):
    data_path = Path(data_path)
    save_path = Path(save_path)
    os.chmod(save_path, stat.S_IRWXO)
    files = save_path.glob("*.pkl")

    for file in files:
        file_name = file.name[:13]
        with open(file, 'rb') as f:
            data = pickle.load(f)
            org_size = data['original_size_of_raw_data']
            org_origin = data['itk_origin']
            org_spacing = data['itk_spacing']
            org_direction = data['itk_direction']
            
            crop_size = data['size_after_cropping']
            crop_bbox = data['crop_bbox']

            resample_spacing = data['spacing_after_resampling']
            resample_size = data['size_after_resampling']

            mask = sitk.ReadImage(str(data_path / f"{file_name}.nii.gz"))
            mask_size = mask.GetSize()
            mask_origin = mask.GetOrigin()
            mask_spacing = mask.GetSpacing()
            mask_direction = mask.GetDirection()

            # crop the mask
            roi_index = (crop_bbox[2][0], crop_bbox[1][0], crop_bbox[0][0])
            roi_size = (crop_size[2], crop_size[1], crop_size[0])

            mask_crop = sitk.RegionOfInterest(mask, roi_size, roi_index)

            # resample the mask
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(mask_crop)
            resampler.SetOutputSpacing((resample_spacing[2], resample_spacing[1], resample_spacing[0]))
            resampler.SetOutputDirection(org_direction)
            resampler.SetOutputOrigin(org_origin)
            resampler.SetSize((resample_size[2], resample_size[1], resample_size[0]))
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            mask_resample = resampler.Execute(mask_crop)

            # save the mask as npz
            mask_npz = sitk.GetArrayFromImage(mask_resample)
            np.savez_compressed(save_path / f"{file_name}_seg.npz", data = mask_npz)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/workdir/results/nnUNet/3d_fullres/Task990_prostate_zonal_Seg/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/predictions_post/')
    parser.add_argument('--save_path', type=str, default='/workdir/nnUNet_preprocessed/Task2302_z-nnmnet/nnUNetData_plans_v2.1_stage0/')
    args = parser.parse_args()
                
    prepare_zonal_mask_npz(
        data_path=args.data_path,
        save_path=args.save_path
    )
