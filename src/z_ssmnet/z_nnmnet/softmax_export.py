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

import os
import pickle
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
from picai_eval.image_utils import read_prediction

from nnunet.preprocessing.preprocessing import (get_do_separate_z,
                                                get_lowres_axis,
                                                resample_data_or_seg)


# adapted from https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/inference/segmentation_export.py
def save_softmax_nifti_from_softmax(segmentation_softmax: Union[str, np.ndarray], out_fname: str,
                                    properties_dict: dict, order: int = 1,
                                    region_class_order: Tuple[Tuple[int]] = None,
                                    seg_postprogess_fn: callable = None, seg_postprocess_args: tuple = None,
                                    resampled_npz_fname: str = None,
                                    non_postprocessed_fname: str = None, force_separate_z: bool = None,
                                    interpolation_order_z: int = 0, verbose: bool = True):
    """
    This is a utility for writing segmentations to nifty and npz. It requires the data to have been preprocessed by
    GenericPreprocessor because it depends on the property dictionary output (dct) to know the geometry of the original
    data. segmentation_softmax does not have to have the same size in pixels as the original data, it will be
    resampled to match that. This is generally useful because the spacings our networks operate on are most of the time
    not the native spacings of the image data.
    If seg_postprogess_fn is not None then seg_postprogess_fnseg_postprogess_fn(segmentation, *seg_postprocess_args)
    will be called before nifty export
    There is a problem with python process communication that prevents us from communicating objects
    larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
    communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long
    enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
    patching system python code.) We circumvent that problem here by saving softmax_pred to a npy file that will
    then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
    filename or np.ndarray for segmentation_softmax and will handle this automatically
    :param segmentation_softmax:
    :param out_fname:
    :param properties_dict:
    :param order:
    :param region_class_order:
    :param seg_postprogess_fn:
    :param seg_postprocess_args:
    :param resampled_npz_fname:
    :param non_postprocessed_fname:
    :param force_separate_z: if None then we dynamically decide how to resample along z, if True/False then always
    /never resample along z separately. Do not touch unless you know what you are doing
    :param interpolation_order_z: if separate z resampling is done then this is the order for resampling in z
    :param verbose:
    :return:
    """
    if verbose: print("force_separate_z:", force_separate_z, "interpolation order:", order)

    if isinstance(segmentation_softmax, str):
        assert os.path.isfile(segmentation_softmax), "If isinstance(segmentation_softmax, str) then " \
                                                     "isfile(segmentation_softmax) must be True"
        # del_file = deepcopy(segmentation_softmax)  # <- skipped this
        if segmentation_softmax.endswith('.npy'):
            segmentation_softmax = np.load(segmentation_softmax)
        elif segmentation_softmax.endswith('.npz'):
            segmentation_softmax = np.load(segmentation_softmax)['softmax']
        # os.remove(del_file)  # <- skipped this

    # first resample, then put result into bbox of cropping, then save
    current_shape = segmentation_softmax.shape
    shape_original_after_cropping = properties_dict.get('size_after_cropping')
    shape_original_before_cropping = properties_dict.get('original_size_of_raw_data')
    # current_spacing = dct.get('spacing_after_resampling')
    # original_spacing = dct.get('original_spacing')

    if np.any([i != j for i, j in zip(np.array(current_shape[1:]), np.array(shape_original_after_cropping))]):
        if force_separate_z is None:
            if get_do_separate_z(properties_dict.get('original_spacing')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(properties_dict.get('original_spacing'))
            elif get_do_separate_z(properties_dict.get('spacing_after_resampling')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(properties_dict.get('spacing_after_resampling'))
            else:
                do_separate_z = False
                lowres_axis = None
        else:
            do_separate_z = force_separate_z
            if do_separate_z:
                lowres_axis = get_lowres_axis(properties_dict.get('original_spacing'))
            else:
                lowres_axis = None

        if lowres_axis is not None and len(lowres_axis) != 1:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False

        if verbose: print("separate z:", do_separate_z, "lowres axis", lowres_axis)
        seg_old_spacing = resample_data_or_seg(segmentation_softmax, shape_original_after_cropping, is_seg=False,
                                               axis=lowres_axis, order=order, do_separate_z=do_separate_z,
                                               order_z=interpolation_order_z)
        # seg_old_spacing = resize_softmax_output(segmentation_softmax, shape_original_after_cropping, order=order)
    else:
        if verbose: print("no resampling necessary")
        seg_old_spacing = segmentation_softmax

    if resampled_npz_fname is not None:
        np.savez_compressed(resampled_npz_fname, softmax=seg_old_spacing.astype(np.float16))
        # this is needed for ensembling if the nonlinearity is sigmoid
        if region_class_order is not None:
            properties_dict['regions_class_order'] = region_class_order
        save_pickle(properties_dict, resampled_npz_fname[:-4] + ".pkl")

    # skip this, we don't want to threshold/argmax softmax output
    # if region_class_order is None:
    #     seg_old_spacing = seg_old_spacing.argmax(0)
    # else:
    #     seg_old_spacing_final = np.zeros(seg_old_spacing.shape[1:])
    #     for i, c in enumerate(region_class_order):
    #         seg_old_spacing_final[seg_old_spacing[i] > 0.5] = c
    #     seg_old_spacing = seg_old_spacing_final
    
    if seg_old_spacing.shape[0] == 2:
        # binary softmax, save foreground only
        seg_old_spacing = seg_old_spacing[1]
    else:
        raise NotImplementedError("only export of binary softmax is implemented")

    bbox = properties_dict.get('crop_bbox')

    if bbox is not None:
        seg_old_size = np.zeros(shape_original_before_cropping, dtype=np.float32)  # <- changed to float32
        for c in range(3):
            bbox[c][1] = np.min((bbox[c][0] + seg_old_spacing.shape[c], shape_original_before_cropping[c]))
        seg_old_size[
            bbox[0][0]:bbox[0][1],
            bbox[1][0]:bbox[1][1],
            bbox[2][0]:bbox[2][1]
        ] = seg_old_spacing
    else:
        seg_old_size = seg_old_spacing

    if seg_postprogess_fn is not None:
        seg_old_size_postprocessed = seg_postprogess_fn(np.copy(seg_old_size), *seg_postprocess_args)
    else:
        seg_old_size_postprocessed = seg_old_size

    seg_resized_itk = sitk.GetImageFromArray(seg_old_size_postprocessed.astype(np.float32))  # <- changed to float32
    seg_resized_itk.SetSpacing(properties_dict['itk_spacing'])
    seg_resized_itk.SetOrigin(properties_dict['itk_origin'])
    seg_resized_itk.SetDirection(properties_dict['itk_direction'])
    sitk.WriteImage(seg_resized_itk, out_fname)

    if (non_postprocessed_fname is not None) and (seg_postprogess_fn is not None):
        seg_resized_itk = sitk.GetImageFromArray(seg_old_size.astype(np.uint8))
        seg_resized_itk.SetSpacing(properties_dict['itk_spacing'])
        seg_resized_itk.SetOrigin(properties_dict['itk_origin'])
        seg_resized_itk.SetDirection(properties_dict['itk_direction'])
        sitk.WriteImage(seg_resized_itk, non_postprocessed_fname)


def convert_cropped_npz_to_original_nifty(
    npz_path: Union[Path, str],
    pkl_path: Optional[Union[Path, str]] = None,
    dst_path: Optional[Union[Path, str]] = None,
    overwrite: bool = False,
) -> None:
    """
    Convert nnUNet's softmax prediction (stored as .npz) to
    the scan's original extent (before cropping).
    Parameters
    ----------
    - npz_path: path to prediction as .npz file
    - pkl_path: path to metadata about nnU-Net preprocessing
        default: npz_path with .pkl instead of .npz
    - dst_path: path to save converted softmax to
        default: npz_path with _softmax.nii.gz instead of .npz
    - overwrite: whether to overwrite an existing softmax prediction
    """
    npz_path = Path(npz_path)
    if pkl_path is None:
        pkl_path = npz_path.with_name(f"{npz_path.stem}.pkl")
    if dst_path is None:
        dst_path = npz_path.with_name(f"{npz_path.stem}_softmax.nii.gz")
    else:
        dst_path = Path(dst_path)
    if not overwrite and dst_path.exists():
        return

    # read prediction and preprocessing info
    # read predictions from all folds
    pred = read_prediction(npz_path)

    # convert to nnUNet's internal softmax format
    pred = np.array([1-pred, pred])

    # read physical properties of current case
    with open(pkl_path, "rb") as fp:
        properties = pickle.load(fp)

    # let nnUNet resample to original physical space
    save_softmax_nifti_from_softmax(
        segmentation_softmax=pred,
        out_fname=str(dst_path),
        properties_dict=properties,
    )