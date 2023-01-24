[← Return to overview](https://github.com/yuanyuan29/Z-SSMNet/tree/master#ssl-pre-training)

# Self Supervised Pre-training

SSL is a general learning framework that relies on surrogate (pretext) tasks that can be formulated using only unsupervised data. A pretext task is designed in a way that solving it requires learning of valuable image representations for the downstream (main) task, which contributes to improve the generalization ability and performance of the model. We introduced image restoration as the pretext task and pre-trained our zonal-aware mesh network in a self-supervised manner.

## Data preparation

1. **Split the Data:** For the SSL part, cross-validation is not needed. We randomly selected 80% of the data from each fold to form the training set, and the rest 20% of the data from each fold form the validation set.
2. **Resampling Spatial Resolution:** The PI-CAI: Public Training and Development Dataset contains MRI scans acquired using seven different scanners, from two vendors, at three centers. Thus, the spatial resolution of its images vary across different patient exams. For instance, in the case of the axial T2W scans, the most common voxel spacing (in mm/voxel) observed is 3.0×0.5×0.5 (43%), followed by 3.6×0.3×0.3 (25%), 3.0×0.342×0.342 (15%) and others (17%). Same as the adaptive settings in [Z-nnMNet](https://github.com/yuanyuan29/Z-SSMNet/blob/master/z-nnmnet.md), we resampled all scans to 3.0×0.5×0.5 mm/voxel.
3. **Cropping to Region-of-Interest:** To extract more informative regions for pre-training, we cropped the images and masks with the region of the prostate's bounding box (based on the generated zonal mask) expanding 2.5cm outward in all directions as the ROI. This allows for tumour outgrowth of the prostate and preserves a clear outline of the prostate relative to adjacent tissues and organs.

Run the command to process the data, the prepared data will be saved into  `/workdir/SSL/data` directory:

```shell
python src/z_ssmnet/ssl/data_preprocessing_zonal.py \
--images_path /workdir/nnUNet_raw_data/Task2302_z-nnmnet/imagesTr/ \
--zonal_mask_path /workdir/results/nnUNet/3d_fullres/Task990_prostate_zonal_Seg/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/predictions_post \
--output_path /workdir/SSL/data \
--splits_path /workdir/nnUNet_raw_data/Task2302_z-nnmnet/splits.json
```

## Create the data generator

We randomly cropped sub-volumes, sized 64×64×16 pixels, from different locations of ROIs to obtain the input 3D cubes. Each sub-folder is named as 'bat_N_s_64x64x16', where *N* denotes the number of cubes extracted from each case. You may select the scale of training samples accordingly based on your resources in hand: larger *N* demands longer learning time and more powerful GPUs/CPUs, while may (or may not) result in a more generic visual representation. We have adopted  *N* =12 in our experiments.

* The processed cubes directory structure

```
generated_cubes/
    |--  bat_12_s_64x64x16_0.npy: cubes extracted from subset0
    |--  bat_12_s_64x64x16_1.npy: cubes extracted from subset1 
    |--  bat_12_s_64x64x16_2.npy: cubes extracted from subset2 
    |--  bat_12_s_64x64x16_3.npy: cubes extracted from subset3 
    |--  bat_12_s_64x64x16_4.npy: cubes extracted from subset4 
    |--  bat_12_s_64x64x16_5.npy: cubes extracted from subset5 
```

Extract 3D cubes from the data by running the script below. The extracted 3D cubes will be saved into `/workdir/SSL/generated_cubes` directory.

```shell
for subset in `seq 0 5`
do
python -W ignore src/z_ssmnet/ssl/infinite_generator_3D_picai_zonal.py \
--fold $subset \
--scale 12 \
--data /dataset_path \
--save /generated_cubes_path
done
```

## Pre-train the Z-SSMNet

```shell
docker run --cpus=8 --memory=64gb --shm-size=64gb --gpus='"device=0"' --rm \
-v /path/to/workdir:/workdir \
-v /path/to/Z-SSMNet:/tmp \
yuanyuan29/z-ssmnet:latest python3 /tmp/src/z_ssmnet/ssl/pretrain/ssl_mnet_zonal.py
```

The pre-trained `Z-SSMNet` will be saved at `/workdir/SSL/pretrained_weights/ssl_mnet_zonal.model`.

## References

[[1]](https://arxiv.org/abs/2205.04846) Z. Dong, Y. He, X. Qi, Y. Chen, H. Shu, J.-L. Coatrieux, G. Yang, and S. Li, “MNet: Rethinking 2D/3D Networks for Anisotropic Medical Image Segmentation,” arXiv:2205.04846, 2022.

[[2]](https://www.sciencedirect.com/science/article/pii/S1361841520302048) Z. Zhou, V. Sodha, J. Pang, M. B. Gotway, and J. Liang, “Models Genesis,” Med Image Anal, vol. 67, pp. 101840, Jan, 2021.
