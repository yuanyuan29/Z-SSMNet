# This code is adapted from https://github.com/MrGiovanni/ModelsGenesis/blob/master/infinite_generator_3D.py. 
# The original code is licensed under the attached LICENSE (https://github.com/yuanyuan29/Z-SSMNet/blob/master/src/z_ssmnet/ssl/LICENSE).

#!/usr/bin/env python
# coding: utf-8

"""
for subset in `seq 0~5`
do
python -W ignore infinite_generator_3D_picai_zonal.py \
--fold $subset \
--scale 12 \
--data /dataset_path \
--save /generated_cubes_path
done
"""

import warnings
warnings.filterwarnings('ignore')
import os

import sys
import random
import numpy as np
import SimpleITK as sitk

from tqdm import tqdm
from optparse import OptionParser
from glob import glob


def make_cubes():
    sys.setrecursionlimit(40000)

    parser = OptionParser()
    parser.add_option("--fold", dest="fold", help="fold of subset", default=None, type="int")
    parser.add_option("--input_rows", dest="input_rows", help="input rows", default=64, type="int")
    parser.add_option("--input_cols", dest="input_cols", help="input cols", default=64, type="int")
    parser.add_option("--input_deps", dest="input_deps", help="input deps", default=16, type="int")
    parser.add_option("--crop_rows", dest="crop_rows", help="crop rows", default=64, type="int")
    parser.add_option("--crop_cols", dest="crop_cols", help="crop cols", default=64, type="int")
    parser.add_option("--data", dest="data", help="the directory of picai dataset", default="/workdir/SSL/data", type="string")
    parser.add_option("--save", dest="save", help="the directory of processed 3D cubes", default="/workdir/SSL/generated_cubes", type="string")
    parser.add_option("--scale", dest="scale", help="scale of the generator", default=12, type="int")
    (options, args) = parser.parse_args()
    fold = options.fold

    seed = 1
    random.seed(seed)

    assert options.data is not None
    assert options.save is not None
    assert options.fold >= 0 and options.fold <= 5

    if not os.path.exists(options.save):
        os.makedirs(options.save)

    class setup_config():
        adc_max = 3000.0
        adc_min = 0.0
        
        def __init__(self, 
                    input_rows=None, 
                    input_cols=None,
                    input_deps=None,
                    crop_rows=None, 
                    crop_cols=None,
                    len_border=None,
                    len_border_z=None,
                    scale=None,
                    DATA_DIR=None,
                    train_fold=[0,1,2,3,4],
                    valid_fold=[5],
                    ):
            self.input_rows = input_rows
            self.input_cols = input_cols
            self.input_deps = input_deps
            self.crop_rows = crop_rows
            self.crop_cols = crop_cols
            self.len_border = len_border
            self.len_border_z = len_border_z
            self.scale = scale
            self.DATA_DIR = DATA_DIR
            self.train_fold = train_fold
            self.valid_fold = valid_fold


        def display(self):
            """Display Configuration values."""
            print("\nConfigurations:")
            for a in dir(self):
                if not a.startswith("__") and not callable(getattr(self, a)):
                    print("{:30} {}".format(a, getattr(self, a)))
            print("\n")


    config = setup_config(input_rows=options.input_rows,
                        input_cols=options.input_cols,
                        input_deps=options.input_deps,
                        crop_rows=options.crop_rows,
                        crop_cols=options.crop_cols,
                        scale=options.scale,
                        len_border=0,
                        len_border_z=0,
                        DATA_DIR=options.data,
                        )
    config.display()

    def infinite_generator_from_one_volume(config, t2_array, adc_array, dwi_array, seg_array):
        size_x, size_y, size_z = t2_array.shape
    
        t2_array = 1.0 * (t2_array - np.min(t2_array)) / (np.max(t2_array)  - np.min(t2_array))
        adc_array[adc_array < config.adc_min] = config.adc_min
        adc_array[adc_array > config.adc_max] = config.adc_max
        adc_array = 1.0 * (adc_array - config.adc_min) / (config.adc_max - config.adc_min)
        dwi_array  = 1.0 * (dwi_array - np.min(dwi_array)) / (np.max(dwi_array) - np.min(dwi_array))

        slice_set = np.zeros((config.scale, 4, config.input_rows, config.input_cols, config.input_deps), dtype=float)
        
        num_pair = 0
        cnt = 0
        while True:
            cnt += 1
            if cnt > 50 * config.scale and num_pair == 0:
                return None
            elif cnt > 50 * config.scale and num_pair > 0:
                return np.array(slice_set[:num_pair])

            start_x = random.randint(0+config.len_border, size_x-config.crop_rows-1-config.len_border)
            start_y = random.randint(0+config.len_border, size_y-config.crop_cols-1-config.len_border)
            start_z = random.randint(0+config.len_border_z, size_z-config.input_deps-1-config.len_border_z)
            
            t2_crop_window = t2_array[start_x : start_x+config.crop_rows,
                                    start_y : start_y+config.crop_cols,
                                    start_z : start_z+config.input_deps,
                                    ]
            
            adc_crop_window = adc_array[start_x : start_x+config.crop_rows,
                                        start_y : start_y+config.crop_cols,
                                        start_z : start_z+config.input_deps,
                                    ]

            dwi_crop_window = dwi_array[start_x : start_x+config.crop_rows,
                                        start_y : start_y+config.crop_cols,
                                        start_z : start_z+config.input_deps,
                                    ]
            seg_crop_window = seg_array[start_x : start_x+config.crop_rows,
                                        start_y : start_y+config.crop_cols, 
                                        start_z : start_z+config.input_deps,
                                    ]

            crop_window = np.stack((t2_crop_window, adc_crop_window, dwi_crop_window, seg_crop_window), axis=0)        
            slice_set[num_pair] = crop_window
            
            num_pair += 1
            if num_pair == config.scale:
                break
                
        return np.array(slice_set)


    def get_self_learning_data(fold, config):
        slice_set = []
        for index_subset in fold:
            picai_subset_path = os.path.join(config.DATA_DIR, "subset"+str(index_subset))
            file_list = glob(os.path.join(picai_subset_path, "*_0000.nii.gz"))
            
            for img_file in tqdm(file_list): 
                print(img_file)          
                t2_img = sitk.ReadImage(img_file, sitk.sitkFloat32) 
                t2_array = sitk.GetArrayFromImage(t2_img)
                t2_array = t2_array.transpose(2, 1, 0)

                adc_img = sitk.ReadImage(img_file.replace('_0000.nii.gz', '_0001.nii.gz'), sitk.sitkFloat32) 
                adc_array = sitk.GetArrayFromImage(adc_img)
                adc_array = adc_array.transpose(2, 1, 0)

                dwi_img = sitk.ReadImage(img_file.replace('_0000.nii.gz', '_0002.nii.gz'), sitk.sitkFloat32) 
                dwi_array = sitk.GetArrayFromImage(dwi_img)
                dwi_array = dwi_array.transpose(2, 1, 0)

                seg_img = sitk.ReadImage(img_file.replace('_0000.nii.gz', '.nii.gz'))
                seg_array = sitk.GetArrayFromImage(seg_img)
                seg_array = seg_array.transpose(2, 1, 0)

                x = infinite_generator_from_one_volume(config, t2_array, adc_array, dwi_array, seg_array)
                if x is not None:
                    slice_set.extend(x)
                
        return np.array(slice_set)


    print(">> Fold {}".format(fold))
    cube = get_self_learning_data([fold], config)
    print("cube: {} | {:.2f} ~ {:.2f}".format(cube.shape, np.min(cube), np.max(cube)))
    np.save(os.path.join(options.save, 
                        "bat_"+str(config.scale)+"_s"+
                        "_"+str(config.input_rows)+
                        "x"+str(config.input_cols)+
                        "x"+str(config.input_deps)+
                        "_"+str(fold)+".npy"), 
            cube,
        )


if __name__ == "__main__":
    make_cubes()
