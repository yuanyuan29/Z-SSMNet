[← Return to overview](https://github.com/yuanyuan29/Z-SSMNet/tree/master#zonal-segmentation)

# Zonal Segmentation

Prostate area is consist of peripharal zone (PZ), transition zone (TZ), central zone (CZ) and anterior fibromuscular stroma (AFS). Prostate cancer (PCa) lesions located in different zones have different characteristics. Moreover, approximately 70%-75% of PCa originate in the PZ and 20%-30% in the TZ [[1]](https://www.sciencedirect.com/science/article/pii/S0302283815008489). In this work, we trained a standard `3D nnU-Net`[ [2]](https://www.nature.com/articles/s41592-020-01008-z) with external public datasets to generate binary prostate zonal anatomy masks (peripheral and rest (TZ, CZ, AFS) of the gland) as additional input information to guide the network to learn region-specific knowledge useful for clinically significant PCa (csPCa) detection and diagnosis. The implementation steps are as follows:

## Installation

The `nnunet` module is installed within the `z-ssmnet` Docker container. For predicting the zonal masks of the images based on our trained model, no additional installation is required. We use the trained model by default.

If you want to retrain the model on your own data, please install the `nnunet` module by following the [official documentation](https://github.com/MIC-DKFZ/nnUNet#installation). Note that if you want to use the `nnunet` installed in our Docker container to retrain the model, you need to comment out the part of the `Dockerfile` that modifies the original `nnunet.`

## Usage of the trained 3D nnU-Net

The trained `nnU-Net `model `"model_final_checkpoint.model"` is saved to the "`workdir/results/nnUNet/3d_fullres/Task990_prostate_zonal_Seg/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0`" directory.

* Prepare the PI-CAI dataset with T2W and ADC images for zonal segmentation

  ```bash
  python src/z_ssmnet/zonal_segmentation/data_preparation.py --images_path /workdir/nnUNet_raw_data/Task2302_z-nnmnet/imagesTr/ --images_zonal_path /workdir/nnUNet_raw_data/Task2302_z-nnmnet/images_zonal
  ```
* Generate the zonal masks

```bash
docker run --cpus=8 --memory=28gb --gpus='"device=0"' --rm \
-v /path/to/images_zonal:/input/images \
-v /path/to/workdir/results:/workdir/results \
-v /path/to/trained model/predictions/:/output/predictions \
yuanyuan29/z-ssmnet:latest nnunet predict Task990_prostate_zonal_Seg \
--trainer nnUNetTrainerV2 \
--fold 0 --checkpoint model_final_checkpoint \
--results /workdir/results \
--input /input/images/ \
--output /output/predictions 

```

To match the revision in `nnUNetTrainer.py` in our Docker container, we manually revise the value of `data['plans']['num_modalities']` to `-1` in `/workdir/results/nnUNet/3d_fullres/Task990_prostate_zonal_Seg/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/model_final_checkpoint.model.pkl`. If you use the original nnU-Net, please return the value back to 2 before prediction.

* Post-processing to remove noisy markers on the zonal masks

```bash
python src/z_ssmnet/zonal_segmentation/zonal_mask_postprocessing.py --zonal_mask_dir /workdir/results/nnUNet/3d_fullres/Task990_prostate_zonal_Seg/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/predictions/ --zonal_mask_post_dir /workdir/results/nnUNet/3d_fullres/Task990_prostate_zonal_Seg/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/predictions_post

```

zonal segmentation completed!

## Train nnU-Net from your own dataset

Please make sure the images of T2W and ADC modalities are included in your dataset and the manual-labelled zonal masks are offered. The data needs to be transfered to [nnUNet raw data archive](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md#dataset-conversion-instructions) format. Just follow the [official documentation](https://github.com/MIC-DKFZ/nnUNet#usage) to train the model.

## Dataset

The public datasets used in our model training:

1. ProstateX: [https://github.com/rcuocolo/PROSTATEx_masks](https://github.com/rcuocolo/PROSTATEx_masks)
2. Prostate158: [https://github.com/kbressem/prostate158](https://github.com/kbressem/prostate158)
3. Medical Segmentation Decathlon (MSD)_Prostate: [http://medicaldecathlon.com](http://medicaldecathlon.com)

## References

[[1]](https://www.sciencedirect.com/science/article/pii/S0302283815008489) J. C. Weinreb, J. O. Barentsz, P. L. Choyke, F. Cornud, M. A. Haider, K. J. Macura, D. Margolis, M. D. Schnall, F. Shtern, C. M. Tempany, H. C. Thoeny, and S. Verma, “PI-RADS Prostate Imaging - Reporting and Data System: 2015, Version 2,” European Urology, vol. 69, no. 1, pp. 16-40, Jan, 2016.

[[2]](https://www.nature.com/articles/s41592-020-01008-z) F. Isensee, P. F. Jaeger, S. A. A. Kohl, J. Petersen, and K. H. Maier-Hein, “nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation,” Nature Methods, vol. 18, no. 2, pp. 203-+, Feb, 2021.
