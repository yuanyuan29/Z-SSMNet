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
import shutil
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--images_path', type=str, default="/workdir/nnUNet_raw_data/Task2302_z-nnmnet/imagesTr")
parser.add_argument('--images_zonal_path', type=str, default="/workdir/nnUNet_raw_data/Task2302_z-nnmnet/images_zonal")
args = parser.parse_args()

if not os.path.exists(args.images_zonal_path):
    os.makedirs(args.images_zonal_path)

cases = glob.glob(os.path.join(args.images_path, "*_0000.nii.gz"))
cases = [os.path.basename(case) for case in cases]

for case in cases:
    t2_path = os.path.join(args.images_path, case)
    adc_path = os.path.join(args.images_path, case.replace("_0000.nii.gz", "_0001.nii.gz"))
    shutil.copyfile(t2_path, os.path.join(args.images_zonal_path, case))
    shutil.copyfile(adc_path, os.path.join(args.images_zonal_path, case.replace("_0000.nii.gz", "_0001.nii.gz")))

print("well done")
