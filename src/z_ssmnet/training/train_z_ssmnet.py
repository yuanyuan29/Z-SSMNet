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


def main(taskname="Task2302_z-nnmnet"):
    """Train Z-SSMNet semi-supervised model."""
    parser = argparse.ArgumentParser()

    # input data and model directories
    parser.add_argument('--workdir', type=str, default="/workdir")
    parser.add_argument('--preprocesseddir', type=str, default=os.environ.get('SM_CHANNEL_PREPROCESSED', "/input/preprocessed"))
    parser.add_argument('--pretrainedweightsdir', type=str, default=os.environ.get('SM_CHANNEL_PRETRAINED', "/input/pretrained"))
    parser.add_argument('--outputdir', type=str, default=os.environ.get('SM_MODEL_DIR', "/output"))
    parser.add_argument('--checkpointsdir', type=str, default="/checkpoints")
    parser.add_argument('--nnUNet_n_proc_DA', type=int, default=None)
    parser.add_argument('--folds', type=int, nargs="+", default=(0, 1, 2, 3, 4),
                        help="Folds to train. Default: 0 1 2 3 4")

    args, _ = parser.parse_known_args()

    # paths
    workdir = Path(args.workdir)
    output_dir = Path(args.outputdir)
    checkpoints_dir = Path(args.checkpointsdir)
    preprocessed_dir = Path(args.preprocesseddir)
    pretrainedweights_dir = Path(args.pretrainedweightsdir)

    workdir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # set environment variables
    os.environ["prepdir"] = (preprocessed_dir / "nnUNet_preprocessed").as_posix()
    if args.nnUNet_n_proc_DA is not None:
        os.environ["nnUNet_n_proc_DA"] = str(args.nnUNet_n_proc_DA)

    # descibe input data
    print(f"workdir: {workdir}")
    print(f"checkpoints_dir: {checkpoints_dir}")
    print(f"preprocessed_dir: {preprocessed_dir}")
    print(f"pretrainedweights_dir: {pretrainedweights_dir}")
    print(f"output_dir: {output_dir}")

    # Train models
    for fold in args.folds:
        print(f"Training fold {fold}...")
        cmd = [
            "nnunet", "plan_train", str(taskname), workdir.as_posix(),
            "--results", checkpoints_dir.as_posix(),
            "--trainer", "myTrainer_zonal",
            "--fold", str(fold),
            "--custom_split", os.path.join(os.environ["prepdir"], taskname, "splits_final.json"),
            "--kwargs=--disable_validation_inference",
            "--use_compressed_data",
            "--pretrained_weights", (pretrainedweights_dir / "ssl_mnet_zonal.model").as_posix(),
        ]
        check_call(cmd)

    # Export trained models
    results_dir = checkpoints_dir / f"nnUNet/3d_fullres/{taskname}/myTrainer_zonal__nnUNetPlansv2.1"
    export_dir = output_dir / f"picai_nnunet_gc_algorithm/results/nnUNet/3d_fullres/{taskname}/myTrainer_zonal__nnUNetPlansv2.1"
    for checkpoint in ["model_best", "model_final_checkpoint"]:
        for fold in args.folds:
            src = results_dir / f"fold_{fold}/{checkpoint}.model"
            dst = export_dir / f"fold_{fold}/{checkpoint}.model"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)

            src = results_dir / f"fold_{fold}/{checkpoint}.model.pkl"
            dst = export_dir / f"fold_{fold}/{checkpoint}.model.pkl"
            shutil.copyfile(src, dst)

    shutil.copyfile(results_dir / "plans.pkl", export_dir / "plans.pkl")


if __name__ == '__main__':
    main()
