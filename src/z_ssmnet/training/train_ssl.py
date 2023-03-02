#!/opt/conda/bin/python

#  Copyright 2022 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import argparse
import os
import shutil
from pathlib import Path
from subprocess import check_call

from z_ssmnet.ssl.pretrain.ssl_mnet_zonal import pretrain


def main(taskname="Task2302_z-nnmnet"):
    """Pretrain nnU-Net model."""
    parser = argparse.ArgumentParser()

    # input data and model directories
    parser.add_argument('--workdir', type=str, default="/workdir")
    parser.add_argument('--imagesdir', type=str, default=os.environ.get('SM_CHANNEL_IMAGES', "/input/images"))
    parser.add_argument('--labelsdir', type=str, default=os.environ.get('SM_CHANNEL_LABELS', "/input/picai_labels"))
    parser.add_argument('--preprocesseddir', type=str, default=os.environ.get('SM_CHANNEL_PREPROCESSED', "/input/preprocessed"))
    parser.add_argument('--outputdir', type=str, default=os.environ.get('SM_MODEL_DIR', "/output"))
    parser.add_argument('--checkpointsdir', type=str, default="/checkpoints")
    parser.add_argument('--folds', type=int, nargs="+", default=(0, 1, 2, 3, 4),
                        help="Folds to train. Default: 0 1 2 3 4")

    args, _ = parser.parse_known_args()

    # paths
    workdir = Path(args.workdir)
    images_dir = Path(args.imagesdir)
    labels_dir = Path(args.labelsdir)
    output_dir = Path(args.outputdir)
    checkpoints_dir = Path(args.checkpointsdir)
    preprocessed_dir = Path(args.preprocesseddir)

    workdir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # descibe input data
    print(f"workdir: {workdir}")
    print(f"images_dir: {images_dir}")
    print(f"labels_dir: {labels_dir}")
    print(f"output_dir: {output_dir}")

    # Pretrain model
    pretrain(
        model_dir=checkpoints_dir / "SSL/pretrained_weights",
        data_dir=preprocessed_dir / "SSL/generated_cubes",
    )

    # Export trained model
    shutil.copytree(checkpoints_dir / "SSL/pretrained_weights", output_dir / "SSL/pretrained_weights")


if __name__ == '__main__':
    main()
