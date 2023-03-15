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

import os
from pathlib import Path
from typing import Union
import SimpleITK as sitk
import numpy as np
import argparse
import glob
import random
import json
from functools import reduce
from tqdm import tqdm

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
    

def infinite_generator_from_one_volume(
    t2_array: np.ndarray,
    adc_array: np.ndarray,
    dwi_array: np.ndarray,
    seg_array: np.ndarray,
    scale: int = 12,
    input_rows: int = 64,
    input_cols: int = 64,
    input_deps: int = 16,
    crop_rows: int = 64,
    crop_cols: int = 64,
    len_border: int = 0,
    len_border_z: int = 0,
):
    adc_max = 3000.0
    adc_min = 0.0

    size_x, size_y, size_z = t2_array.shape
 
    t2_array = 1.0 * (t2_array - np.min(t2_array)) / (np.max(t2_array)  - np.min(t2_array))
    adc_array[adc_array < adc_min] = adc_min
    adc_array[adc_array > adc_max] = adc_max
    adc_array = 1.0 * (adc_array - adc_min) / (adc_max - adc_min)
    dwi_array  = 1.0 * (dwi_array - np.min(dwi_array)) / (np.max(dwi_array) - np.min(dwi_array))

    slice_set = np.zeros((scale, 4, input_rows, input_cols, input_deps), dtype=float)
    
    num_pair = 0
    cnt = 0
    while True:
        cnt += 1
        if cnt > 50 * scale and num_pair == 0:
            return None
        elif cnt > 50 * scale and num_pair > 0:
            return np.array(slice_set[:num_pair])

        start_x = random.randint(0+len_border, size_x-crop_rows-1-len_border)
        start_y = random.randint(0+len_border, size_y-crop_cols-1-len_border)
        start_z = random.randint(0+len_border_z, size_z-input_deps-1-len_border_z)
        
        t2_crop_window = t2_array[start_x : start_x+crop_rows,
                                  start_y : start_y+crop_cols,
                                  start_z : start_z+input_deps,
                                 ]
        
        adc_crop_window = adc_array[start_x : start_x+crop_rows,
                                    start_y : start_y+crop_cols,
                                    start_z : start_z+input_deps,
                                   ]

        dwi_crop_window = dwi_array[start_x : start_x+crop_rows,
                                    start_y : start_y+crop_cols,
                                    start_z : start_z+input_deps,
                                   ]
        seg_crop_window = seg_array[start_x : start_x+crop_rows,
                                    start_y : start_y+crop_cols, 
                                    start_z : start_z+input_deps,
                                   ]

        crop_window = np.stack((t2_crop_window, adc_crop_window, dwi_crop_window, seg_crop_window), axis=0)        
        slice_set[num_pair] = crop_window
        
        num_pair += 1
        if num_pair == scale:
            break
            
    return np.array(slice_set)


def data_preprocessing_zonal(
    images_path: Union[Path, str],
    zonal_mask_path: Union[Path, str],
    splits_path: Union[Path, str],
    output_train_path: Union[Path, str],
    output_val_path: Union[Path, str],
    scale: int = 12,
    input_rows: int = 64,
    input_cols: int = 64,
    input_deps: int = 16,
    crop_rows: int = 64,
    crop_cols: int = 64,
    len_border: int = 0,
    len_border_z: int = 0,
):
    if not os.path.exists(output_train_path):
        os.makedirs(output_train_path)
    if not os.path.exists(output_val_path):
        os.makedirs(output_val_path)

    # split data
    random.seed(1)
    train_ids = []
    val_ids = []
    with open(splits_path, 'r') as f:
        splits = json.load(f)
        for i in range(5):
            val = splits[i]["val"]
            random.shuffle(val)
            train_ids.append(val[:int(len(val)*0.8)])
            val_ids.append(val[int(len(val)*0.8):])

    train_ids = reduce(lambda x,y: x.extend(y) or x, train_ids)
    val_ids = reduce(lambda x,y: x.extend(y) or x, val_ids)

    for id in tqdm(train_ids):
        t2 = sitk.ReadImage(os.path.join(images_path, id + '_0000.nii.gz'), sitk.sitkFloat32)
        adc = sitk.ReadImage(os.path.join(images_path, id + '_0001.nii.gz'), sitk.sitkFloat32)
        dwi = sitk.ReadImage(os.path.join(images_path, id + '_0002.nii.gz'), sitk.sitkFloat32)
        seg = sitk.ReadImage(os.path.join(zonal_mask_path, id + '.nii.gz'))

        # image resample
        t2 = resample(t2, mask=False)
        adc = resample(adc, mask=False)
        dwi = resample(dwi, mask=False)
        seg = resample(seg, mask=True)

        # crop the ROI 
        t2, adc, dwi, seg = crop(t2, adc, dwi, seg)  

        # extract sub-volumes
        t2_array = sitk.GetArrayFromImage(t2)
        t2_array = t2_array.transpose(2, 1, 0)

        adc_array = sitk.GetArrayFromImage(adc)
        adc_array = adc_array.transpose(2, 1, 0)

        dwi_array = sitk.GetArrayFromImage(dwi)
        dwi_array = dwi_array.transpose(2, 1, 0)

        seg_array = sitk.GetArrayFromImage(seg)
        seg_array = seg_array.transpose(2, 1, 0)

        volumes = infinite_generator_from_one_volume(
            t2_array, adc_array, dwi_array, seg_array,
            scale, input_rows, input_cols, input_deps,
            crop_rows, crop_cols, len_border, len_border_z,
        )

        for i in range(len(volumes)):
            np.save(os.path.join(output_train_path, id + '_' + "%02d" % i + '.npy'), volumes[i])

    for id in tqdm(val_ids):
        t2 = sitk.ReadImage(os.path.join(images_path, id + '_0000.nii.gz'), sitk.sitkFloat32)
        adc = sitk.ReadImage(os.path.join(images_path, id + '_0001.nii.gz'), sitk.sitkFloat32)
        dwi = sitk.ReadImage(os.path.join(images_path, id + '_0002.nii.gz'), sitk.sitkFloat32)
        seg = sitk.ReadImage(os.path.join(zonal_mask_path, id + '.nii.gz'))

        # image resample
        t2 = resample(t2, mask=False)
        adc = resample(adc, mask=False)
        dwi = resample(dwi, mask=False)
        seg = resample(seg, mask=True)

        # crop the ROI 
        t2, adc, dwi, seg = crop(t2, adc, dwi, seg)  

        # extract sub-volumes
        t2_array = sitk.GetArrayFromImage(t2)
        t2_array = t2_array.transpose(2, 1, 0)

        adc_array = sitk.GetArrayFromImage(adc)
        adc_array = adc_array.transpose(2, 1, 0)

        dwi_array = sitk.GetArrayFromImage(dwi)
        dwi_array = dwi_array.transpose(2, 1, 0)

        seg_array = sitk.GetArrayFromImage(seg)
        seg_array = seg_array.transpose(2, 1, 0)

        volumes = infinite_generator_from_one_volume(
            t2_array, adc_array, dwi_array, seg_array,
            scale, input_rows, input_cols, input_deps,
            crop_rows, crop_cols, len_border, len_border_z,
        )

        for i in range(len(volumes)):
            np.save(os.path.join(output_val_path, id + '_' + "%02d" % i + '.npy'), volumes[i])

    print('well done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', type=str, default='/workdir/nnUNet_raw_data/Task2302_z-nnmnet/imagesTr')
    parser.add_argument('--zonal_mask_path', type=str, default='/workdir/results/nnUNet/3d_fullres/Task990_prostate_zonal_Seg/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/predictions_post')
    parser.add_argument('--output_train_path', type=str, default='/workdir/SSL/generated_cubes/train')
    parser.add_argument('--output_val_path', type=str, default='/workdir/SSL/generated_cubes/val')
    parser.add_argument('--splits_path', type=str, default='/workdir/nnUNet_raw_data/Task2302_z-nnmnet/splits.json')

    parser.add_argument("--input_rows", type=int, default=64)
    parser.add_argument("--input_cols", type=int, default=64)
    parser.add_argument("--input_deps", type=int, default=16)
    parser.add_argument("--crop_rows", type=int, default=64)
    parser.add_argument("--crop_cols", type=int, default=64)
    parser.add_argument("--len_border", type=int, default=0)
    parser.add_argument("--len_border_z", type=int, default=0)

    parser.add_argument("--scale", type=int, default=12, help="scale of the generator")

    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    data_preprocessing_zonal(
        images_path=args.images_path,
        zonal_mask_path=args.zonal_mask_path,
        splits_path=args.splits_path,
        output_train_path=args.output_train_path,
        output_val_path=args.output_val_path,
        scale=args.scale,
        input_rows=args.input_rows,
        input_cols=args.input_cols,
        input_deps=args.input_deps,
        crop_rows=args.crop_rows,
        crop_cols=args.crop_cols,
        len_border=args.len_border,
        len_border_z=args.len_border_z,
    )
