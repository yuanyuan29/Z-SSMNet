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
import json
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
from picai_eval import evaluate_folder
from picai_prep.preprocessing import crop_or_pad
from report_guided_annotation import extract_lesion_candidates

from picai_baseline.nnunet.softmax_export import \
    convert_cropped_npz_to_original_nifty
from picai_baseline.splits.picai import valid_splits as picai_pub_valid_splits
from picai_baseline.splits.picai_nnunet import \
    valid_splits as picai_pub_nnunet_valid_splits
from picai_eval.image_utils import read_prediction
import SimpleITK as sitk


try:
    import numpy.typing as npt
except ImportError:  # pragma: no cover
    pass


def extract_lesion_candidates_originated_from_prostate(
    softmax_path: Union[Path, str], 
    zonal_mask_path: Union[Path, str],
    dst_path: Optional[Union[Path, str]] = None,
    overwrite: bool = False,
) -> None:
    """Extract the candidate lesions that is originated from prostate"""
    softmax_path = Path(softmax_path)
    zonal_mask_path = Path(zonal_mask_path)

    if dst_path is None:
        dst_path = softmax_path.parent / softmax_path.name.replace("_softmax.nii.gz", "_detection_map.nii.gz")
    else:
        dst_path = Path(dst_path)
    if not overwrite and dst_path.exists():
        return

    # read softmax and zonal mask
    softmax_array = read_prediction(softmax_path)
    zonal_mask_array = read_prediction(zonal_mask_path)

    # generate the detection map
    detection_map = extract_lesion_candidates(softmax=softmax_array, threshold="dynamic")[0]
    
    # calculate the connected component of the detection map
    binary_detection_map = (detection_map > 0).astype(np.uint8)
    binary_detection_map: sitk.Image = sitk.GetImageFromArray(binary_detection_map)
    binary_detection_map.CopyInformation(sitk.ReadImage(str(softmax_path)))

    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)
    out_mask = cc_filter.Execute(binary_detection_map)
    out_mask_array = sitk.GetArrayFromImage(out_mask)

    num_cc = cc_filter.GetObjectCount()
    for i in range(1, num_cc + 1):
        # select the connected component
        cc = (out_mask_array == i).astype(np.uint8)
        # calculate the intersection with zonal mask
        intersection = np.sum(cc * zonal_mask_array)
        if intersection == 0:
            # remove the connected component that without intersection with zonal mask
            detection_map[cc == 1] = 0
        else:
            continue
    
    # save the detection map
    detection_map: sitk.Image = sitk.GetImageFromArray(detection_map)
    detection_map.CopyInformation(sitk.ReadImage(str(softmax_path)))
    sitk.WriteImage(detection_map, str(dst_path))
    

def evaluate(
    task: str = "Task2302_z-nnmnet",
    trainer: str = "myTrainer_zonal",
    workdir: Union[Path, str] = "/workdir",
    task_dir: Union[Path, str] = "auto",
    checkpoints: str = ["model_best"],
    folds: str = list(range(5)),
    threshold: str = "dynamic",
    metrics_fn: str = "metrics-{checkpoint}-{threshold}-zonal.json",
    splits: str = "picai_pub",
    predictions_folder: str = r"predictions_{checkpoint}",
    labels_folder: Union[Path, str] = "auto",
    zonal_mask_folder: Union[Path, str] = "/workdir/results/nnUNet/3d_fullres/Task990_prostate_zonal_Seg/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/predictions_post/",
    verbose: int = 2,
):
    # input validation
    workdir = Path(workdir)
    zonal_mask_folder = Path(zonal_mask_folder)
    if task_dir == "auto":
        task_dir = workdir / "results" / "nnUNet" / "3d_fullres" / task
    else:
        task_dir = workdir / task_dir
    
    if isinstance(splits, str):
        if splits == "":
            splits = None
        else:
            # select splits
            predefined_splits = {
                "picai_pub": picai_pub_valid_splits,              
                "picai_pub_nnunet": picai_pub_nnunet_valid_splits,
            }
            if splits in predefined_splits:
                splits = predefined_splits[splits]
            else:
                # `splits` should be the path to a json file containing the splits
                print(f"Loading splits from {splits}")

    
    for fold in folds:
        print(f"Evaluating fold {fold}...")

        for checkpoint in checkpoints:
            pred_folder = predictions_folder.replace(r"{checkpoint}", checkpoint)
            softmax_dir = task_dir / f"{trainer}__nnUNetPlansv2.1" / f"fold_{fold}" / pred_folder
            metrics_path = softmax_dir.parent / metrics_fn.replace(r"{checkpoint}", checkpoint).replace(r"{threshold}", threshold).replace("{fold}", str(fold))
 

            if metrics_path.exists():
                print(f"Metrics found at {metrics_path}, skipping..")
                continue
            else:
                print(f"Metrics will be saved to {metrics_path}.")

            original_softmax_prediction_paths = softmax_dir.glob("*.npz")

            if verbose >= 2:
                print(f"Predictions folder: {softmax_dir}")
                original_softmax_prediction_paths = list(original_softmax_prediction_paths)
                print(f"Found {len(original_softmax_prediction_paths)} predictions (e.g., {original_softmax_prediction_paths[0:2]})")

            # pad raw npz predictions to their original extent
            print("Converting nnU-Net predictions to original extent...")
            with Pool() as pool:
                pool.map(
                    func=convert_cropped_npz_to_original_nifty,
                    iterable=original_softmax_prediction_paths,
                )

            # postprocess softmax predictions
            print("Postprocessing softmax predictions...")
            softmax_prediction_paths = list(softmax_dir.glob("*_softmax.nii.gz"))
            for softmax_path in softmax_prediction_paths:
                extract_lesion_candidates_originated_from_prostate(softmax_path, Path(zonal_mask_folder / softmax_path.name.replace("_softmax.nii.gz", ".nii.gz")))

            # select subject_list
            if splits is None:
                subject_list = None
            elif isinstance(splits, dict):
                subject_list = splits[fold]['subject_list']
            elif isinstance(splits, str):
                path = Path(splits.replace(r"{fold}", str(fold)))
                with open(path, "r") as f:
                    splits = json.load(f)
                subject_list = splits['subject_list']
            else:
                raise ValueError(f"Unrecognised splits: {splits}")

            # select labels folder
            if labels_folder == "auto":
                y_true_dir = workdir / "nnUNet_raw_data" / task / "labelsTr"
            else:
                y_true_dir = workdir / labels_folder.replace(r"{fold}", str(fold))
           
            # evaluate
            metrics = evaluate_folder(
                y_det_dir=softmax_dir,
                y_true_dir=y_true_dir,
                subject_list=subject_list,
                pred_extensions=['_detection_map.nii.gz'],
                num_parallel_calls=5,
            )

            # save and show metrics
            metrics.save(metrics_path)
            print(f"Results for checkpoint {checkpoint}:")
            print(metrics)


def main():
    parser = argparse.ArgumentParser(description='Command Line Arguments')
    parser.add_argument("--task", type=str, default="Task2302_z-nnmnet",
                        help="Task name of the Z-nnMNet experiment. Default: Task2302_z-nnmnet")
    parser.add_argument("--trainer", type=str, default="myTrainer_zonal",
                        help="Trainer of the Z-nnMNet experiment. Default: myTrainer_zonal")
    parser.add_argument("--workdir", type=str, default="/workdir",
                        help="Path to the workdir where 'results' and 'nnUNet_raw_data' are stored. Default: /workdir")
    parser.add_argument("--task_dir", type=str, default="auto",
                        help="Path to the task directory (relative to the workdir). Optional, will be constucted for default nnU-Net forlder structure")
    parser.add_argument("--checkpoints", type=str, nargs="+", default=["model_best"],
                        help="Which checkpoints to evaluate. Multiple checkpoints can be passed at once. Default: model_best")
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)),
                        help="Which folds to evaluate. Multiple folds can be evaluated at once. Default: 0, 1, 2, 3, 4  (all)")
    parser.add_argument("--threshold", type=str, default="dynamic",
                        help="Threshold for lesion extraction from softmax predictions. " + \
                             "Use dynamic-fast for quicker evaluation at almost equal performance." + \
                                "Default: dynamic.")
    parser.add_argument("--metrics_fn", type=str, default=r"metrics-{checkpoint}-{threshold}-zonal.json",
                        help=r"Filename to save metrics to. May contain {checkpoint} and {threshold} which are auto-filled. Default: metrics-{checkpoint}-{threshold}-zonal.json")
    parser.add_argument("--splits", type=str, default="picai_pub_nnunet",
                        help="Splits for cross-validation. Available predefined splits: picai_pub, picai_pub_nnunet. Alternatively, provide a path to a json file " +
                             "containing a dictiory with key 'subject_list', where `{fold}` in the path is replaced by " +
                             "the fold. Example: `/workdir/splits/val/fold_{fold}.json`. Default: picai_pub_nnunet.")
    parser.add_argument("--predictions_folder", type=str, default="predictions_{checkpoint}",
                        help=r"Folder with nnU-Net softmax predictions.")
    parser.add_argument("--labels_folder", type=str, default="auto",
                        help="Folder with annotations. Optional, will be constucted to labelsTr of the specified " +
                             "nnU-Net task.")
    parser.add_argument("--zonal_mask_folder", type=str, default="/workdir/results/nnUNet/3d_fullres/Task990_prostate_zonal_Seg/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/predictions_post/",
                        help="Folder with zonal masks.")
    args = parser.parse_args()

    # evaluate
    evaluate(
        task=args.task,
        trainer=args.trainer,
        workdir=args.workdir,
        task_dir=args.task_dir,
        checkpoints=args.checkpoints,
        folds=args.folds,
        threshold=args.threshold,
        metrics_fn=args.metrics_fn,
        splits=args.splits,
        predictions_folder=args.predictions_folder,
        labels_folder=args.labels_folder,
        zonal_mask_folder=args.zonal_mask_folder,
    )

if __name__ == "__main__":
    main()