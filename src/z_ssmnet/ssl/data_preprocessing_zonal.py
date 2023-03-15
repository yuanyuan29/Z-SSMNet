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
import glob
import json
import os
import random
from functools import reduce

import numpy as np
import SimpleITK as sitk


def resample(vol, mask=True, new_spacing=[0.5, 0.5, 3.0]):
    # Determine current pixel spacing
    spacing = vol.GetSpacing() # x, y, z 
    # Compute new dimensions
    resize_factor = [spacing[i] / new_spacing[i] for i in range(len(spacing))]
    new_real_shape = [vol.GetSize()[i] * resize_factor[i] for i in range(len(spacing))]
    new_shape = np.round(new_real_shape)
    real_resize_factor = [new_shape[i] / vol.GetSize()[i] for i in range(len(new_shape))]
    new_spacing = [spacing[i] / real_resize_factor[i] for i in range(len(spacing))]
    new_shape = [int(i) for i in new_shape]

    # Resample
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_shape)
    resample.SetOutputDirection(vol.GetDirection())
    resample.SetOutputOrigin(vol.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)
    if mask:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)
    new_vol = resample.Execute(vol)

    return new_vol


def crop(t2, adc, dwi, seg):
    seg_array = sitk.GetArrayFromImage(seg)
    seg_array[seg_array > 0]  = 1
    seg_array[seg_array <= 0] = 0
    seg_img = sitk.GetImageFromArray(seg_array)
    seg_img.CopyInformation(seg)
    seg_img = sitk.Cast(seg_img, sitk.sitkUInt8)
    labelFilter = sitk.LabelShapeStatisticsImageFilter()
    labelFilter.Execute(seg_img)
    bbox = labelFilter.GetBoundingBox(1)

    # Crop
    x_start, x_end = max(0, bbox[0]-50), min(bbox[0] + bbox[3] + 50, t2.GetSize()[0])
    y_start, y_end = max(0, bbox[1]-50), min(bbox[1] + bbox[4] + 50, t2.GetSize()[1])
    z_start, z_end = max(0, bbox[2]-50), min(bbox[2] + bbox[5] + 50, t2.GetSize()[2])

    cropFilter = sitk.RegionOfInterestImageFilter()
    cropFilter.SetSize([x_end - x_start, y_end - y_start, z_end - z_start])
    cropFilter.SetIndex([x_start, y_start, z_start])
    cropped_t2 = cropFilter.Execute(t2)
    cropped_adc = cropFilter.Execute(adc)
    cropped_dwi = cropFilter.Execute(dwi)
    cropped_seg = cropFilter.Execute(seg)

    return cropped_t2, cropped_adc, cropped_dwi, cropped_seg
    

def data_preprocessing_zonal(
    images_path,
    zonal_mask_path,
    output_path,
    splits_path,
):
    # split data
    random.seed(1)
    folds = [[] for _ in range(6)] 
    with open(splits_path, 'r') as f:
        splits = json.load(f)
        for i in range(5):
            val = splits[i]["val"]
            random.shuffle(val)
            folds[i].append(val[:int(len(val)*0.8)])
            folds[5].append(val[int(len(val)*0.8):])

    f0 = reduce(lambda x,y: x.extend(y) or x, folds[0])
    f1 = reduce(lambda x,y: x.extend(y) or x, folds[1])
    f2 = reduce(lambda x,y: x.extend(y) or x, folds[2])
    f3 = reduce(lambda x,y: x.extend(y) or x, folds[3])
    f4 = reduce(lambda x,y: x.extend(y) or x, folds[4])
    f5 = reduce(lambda x,y: x.extend(y) or x, folds[5])

    # read images
    images = glob.glob(os.path.join(images_path, '*_0000.nii.gz'))
    for image in images:
        case_id = image.split('/')[-1][:13]
        t2 = sitk.ReadImage(os.path.join(images_path, case_id + '_0000.nii.gz'), sitk.sitkFloat32)
        adc = sitk.ReadImage(os.path.join(images_path, case_id + '_0001.nii.gz'), sitk.sitkFloat32)
        dwi = sitk.ReadImage(os.path.join(images_path, case_id + '_0002.nii.gz'), sitk.sitkFloat32)
        seg = sitk.ReadImage(os.path.join(zonal_mask_path, case_id + '.nii.gz'))

        # image resample
        t2 = resample(t2, mask=False)
        adc = resample(adc, mask=False)
        dwi = resample(dwi, mask=False)
        seg = resample(seg, mask=True)

        # crop the ROI 
        t2, adc, dwi, seg = crop(t2, adc, dwi, seg)  

        # save
        for i in range(6):
            os.makedirs(os.path.join(output_path, 'subset{}'.format(i)), exist_ok=True)

        if case_id in f0:   
            sitk.WriteImage(t2, os.path.join(output_path, 'subset0', case_id + '_0000.nii.gz'))
            sitk.WriteImage(adc, os.path.join(output_path, 'subset0', case_id + '_0001.nii.gz'))
            sitk.WriteImage(dwi, os.path.join(output_path, 'subset0', case_id + '_0002.nii.gz'))
            sitk.WriteImage(seg, os.path.join(output_path, 'subset0', case_id + '.nii.gz'))
        elif case_id in f1:
            sitk.WriteImage(t2, os.path.join(output_path, 'subset1', case_id + '_0000.nii.gz'))
            sitk.WriteImage(adc, os.path.join(output_path, 'subset1', case_id + '_0001.nii.gz'))
            sitk.WriteImage(dwi, os.path.join(output_path, 'subset1', case_id + '_0002.nii.gz'))
            sitk.WriteImage(seg, os.path.join(output_path, 'subset1', case_id + '.nii.gz'))
        elif case_id in f2:
            sitk.WriteImage(t2, os.path.join(output_path, 'subset2', case_id + '_0000.nii.gz'))
            sitk.WriteImage(adc, os.path.join(output_path, 'subset2', case_id + '_0001.nii.gz'))
            sitk.WriteImage(dwi, os.path.join(output_path, 'subset2', case_id + '_0002.nii.gz'))
            sitk.WriteImage(seg, os.path.join(output_path, 'subset2', case_id + '.nii.gz'))
        elif case_id in f3:
            sitk.WriteImage(t2, os.path.join(output_path, 'subset3', case_id + '_0000.nii.gz'))
            sitk.WriteImage(adc, os.path.join(output_path, 'subset3', case_id + '_0001.nii.gz'))
            sitk.WriteImage(dwi, os.path.join(output_path, 'subset3', case_id + '_0002.nii.gz'))
            sitk.WriteImage(seg, os.path.join(output_path, 'subset3', case_id + '.nii.gz'))
        elif case_id in f4:
            sitk.WriteImage(t2, os.path.join(output_path, 'subset4', case_id + '_0000.nii.gz'))
            sitk.WriteImage(adc, os.path.join(output_path, 'subset4', case_id + '_0001.nii.gz'))
            sitk.WriteImage(dwi, os.path.join(output_path, 'subset4', case_id + '_0002.nii.gz'))
            sitk.WriteImage(seg, os.path.join(output_path, 'subset4', case_id + '.nii.gz'))
        elif case_id in f5:
            sitk.WriteImage(t2, os.path.join(output_path, 'subset5', case_id + '_0000.nii.gz'))
            sitk.WriteImage(adc, os.path.join(output_path, 'subset5', case_id + '_0001.nii.gz'))
            sitk.WriteImage(dwi, os.path.join(output_path, 'subset5', case_id + '_0002.nii.gz'))
            sitk.WriteImage(seg, os.path.join(output_path, 'subset5', case_id + '.nii.gz'))

    print('well done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', type=str, default='/workdir/nnUNet_raw_data/Task2302_z-nnmnet/imagesTr')
    parser.add_argument('--zonal_mask_path', type=str, default='/workdir/results/nnUNet/3d_fullres/Task990_prostate_zonal_Seg/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/predictions_post')
    parser.add_argument('--output_path', type=str, default='/workdir/SSL/data')
    parser.add_argument('--splits_path', type=str, default='/workdir/nnUNet_raw_data/Task2302_z-nnmnet/splits.json')
    args = parser.parse_args()

    data_preprocessing_zonal(
        images_path=args.images_path,
        zonal_mask_path=args.zonal_mask_path,
        output_path=args.output_path,
        splits_path=args.splits_path
    )
