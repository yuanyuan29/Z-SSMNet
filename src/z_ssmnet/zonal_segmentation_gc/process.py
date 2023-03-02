import os
import subprocess
from pathlib import Path

import SimpleITK as sitk
from evalutils import SegmentationAlgorithm
from evalutils.validators import (UniqueImagesValidator,
                                  UniquePathIndicesValidator)
from picai_prep.data_utils import atomic_image_write
from picai_prep.preprocessing import (PreprocessingSettings, Sample,
                                      resample_to_reference_scan)

import numpy as np

class MissingSequenceError(Exception):
    """Exception raised when a sequence is missing."""

    def __init__(self, name, folder):
        message = f"Could not find scan for {name} in {folder} (files: {os.listdir(folder)})"
        super().__init__(message)


class MultipleScansSameSequencesError(Exception):
    """Exception raised when multiple scans of the same sequences are provided."""

    def __init__(self, name, folder):
        message = f"Found multiple scans for {name} in {folder} (files: {os.listdir(folder)})"
        super().__init__(message)


def strip_metadata(img: sitk.Image) -> None:
    for key in img.GetMetaDataKeys():
        img.EraseMetaData(key)

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
    
class ProstateZonalSegmentationAlgorithm(SegmentationAlgorithm):
    """
    Wrapper to deploy trained prostate zonal segmentation nnU-Net model from
    https://github.com/yuanyuan29/Z-SSMNet/ as a grand-challenge.org algorithm.
    """

    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )

        # input / output paths for algorithm
        self.input_dirs = [
            "/input/images/transverse-t2-prostate-mri",
            "/input/images/transverse-adc-prostate-mri",
        ]
        self.scan_paths = []
        self.prostate_zonal_segmentation_path = Path("/output/images/prostate-zonal-segmentation/prostate_zonal_mask.mha")

        # input / output paths for nnUNet
        self.nnunet_inp_dir = Path("/opt/algorithm/nnunet/input")
        self.nnunet_out_dir = Path("/opt/algorithm/nnunet/output")
        self.nnunet_results = Path("/opt/algorithm/results")

        # ensure required folders exist
        self.nnunet_inp_dir.mkdir(exist_ok=True, parents=True)
        self.nnunet_out_dir.mkdir(exist_ok=True, parents=True)
        self.prostate_zonal_segmentation_path.parent.mkdir(exist_ok=True, parents=True)

        # input validation for multiple inputs
        scan_glob_format = "*.mha"
        for folder in self.input_dirs:
            file_paths = list(Path(folder).glob(scan_glob_format))
            if len(file_paths) == 0:
                raise MissingSequenceError(name=folder.split("/")[-1], folder=folder)
            elif len(file_paths) >= 2:
                raise MultipleScansSameSequencesError(name=folder.split("/")[-1], folder=folder)
            else:
                # append scan path to algorithm input paths
                self.scan_paths += [file_paths[0]]

    def preprocess_input(self):
        """Preprocess input images to nnUNet Raw Data Archive format"""
        # set up Sample
        sample = Sample(
            scans=[
                sitk.ReadImage(str(path))
                for path in self.scan_paths
            ],
            # settings=PreprocessingSettings(
            #     physical_size=[81.0, 192.0, 192.0],
            #     crop_only=True
            # )
        )

        # perform preprocessing
        sample.preprocess()

        # write preprocessed scans to nnUNet input directory
        for i, scan in enumerate(sample.scans):
            path = self.nnunet_inp_dir / f"scan_{i:04d}.nii.gz"
            atomic_image_write(scan, path)

    # Note: need to overwrite process because of flexible inputs, which requires custom data loading
    def process(self):
        """
        Load bpMRI scans and segment the PZ and TZ of prostate gland
        """
        # perform preprocessing
        self.preprocess_input()

        # perform inference using nnUNet
        self.predict(
            task="Task990_prostate_zonal_Seg",
            trainer="nnUNetTrainerV2",
            checkpoint="model_final_checkpoint",
            folds="0",
            store_probability_maps=False,
        )

        # read binarized prediction
        pred_path = str(self.nnunet_out_dir / "scan.nii.gz")
        pred: sitk.Image = sitk.ReadImage(pred_path)

        # postprocess to remove the noisy labels
        pred = mask_postprocessing(pred)

        # transform prediction to original space
        reference_scan = sitk.ReadImage(str(self.scan_paths[0]))
        pred = resample_to_reference_scan(pred, reference_scan_original=reference_scan)

        # remove metadata to get rid of SimpleITK warning
        strip_metadata(pred)

        # save prediction to output folder
        atomic_image_write(pred, str(self.prostate_zonal_segmentation_path))

    def predict(self, task, trainer="nnUNetTrainerV2", network="3d_fullres",
                checkpoint="model_final_checkpoint", folds="0,1,2,3,4", store_probability_maps=True,
                disable_augmentation=False, disable_patch_overlap=False):
        """
        Use trained nnUNet network to generate segmentation masks
        """

        # Set environment variables
        os.environ['RESULTS_FOLDER'] = str(self.nnunet_results)

        # Run prediction script
        cmd = [
            'nnUNet_predict',
            '-t', task,
            '-i', str(self.nnunet_inp_dir),
            '-o', str(self.nnunet_out_dir),
            '-m', network,
            '-tr', trainer,
            '--num_threads_preprocessing', '2',
            '--num_threads_nifti_save', '1'
        ]

        if folds:
            cmd.append('-f')
            cmd.extend(folds.split(','))

        if checkpoint:
            cmd.append('-chk')
            cmd.append(checkpoint)

        if store_probability_maps:
            cmd.append('--save_npz')

        if disable_augmentation:
            cmd.append('--disable_tta')

        if disable_patch_overlap:
            cmd.extend(['--step_size', '1'])

        subprocess.check_call(cmd)


if __name__ == "__main__":
    ProstateZonalSegmentationAlgorithm().process()
