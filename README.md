# Z-SSMNet: A Zonal-aware Self-Supervised Mesh Network for Prostate Cancer Detection and Diagnosis in Bi-parametric MRI

This repository provides the **official** implementation of the paper "[Z-SSMNet: A Zonal-aware Self-Supervised Mesh Network for Prostate Cancer Detection and Diagnosis in bpMRI](https://arxiv.org/abs/2212.05808)". In this paper, we propose a new Zonal-aware Self-supervised Mesh Network that adaptively fuses multiple 2D/2.5D/3D CNNs to effectively balance representation for sparse inter-slice information and dense intra-slice information in bpMRI. A self-supervised learning (SSL) technique is further introduced to pre-train our network using unlabelled data to learn the generalizable image features. Furthermore, we constrained our network to understand the zonal-specific domain knowledge to improve the diagnosis precision of clinically significant prostate cancer (csPCa). Our model was developed on the [PI-CAI dataset](https://zenodo.org/record/6624726) while participating in the [PI-CAI challenge](https://pi-cai.grand-challenge.org/).

For more information about the **Z-SSMNet**, please read the following paper (arXiv preprint)

```
@article{yuan2022z,
  title={Z-SSMNet: A Zonal-aware Self-Supervised Mesh Network for Prostate Cancer Detection and Diagnosis in bpMRI},
  author={Yuan, Yuan and Ahn, Euijoon and Feng, Dagan and Khadra, Mohamad and Kim, Jinman},
  journal={arXiv preprint arXiv:2212.05808},
  year={2022}
}
```

## Issues

Please feel free to raise any issues you encounter [here](https://github.com/yuanyuan29/Z-SSMNet/issues).

## Installation

`Z-SSMNet` can be pip-installed directly:

```shell
pip install git+https://github.com/yuanyuan29/Z-SSMNet.git
```

Alternatively, `Z-SSMNet` can be installed from source:

```shell
git clone https://github.com/yuanyuan29/Z-SSMNet.git
cd Z-SSMNet
pip install -e .
```

This ensures the scripts are present locally, which enables you to run the provided Python scripts. Additionally, this allows you to modify the offered solutions, due to the `-e` option.

## General Setup

We define setup steps that must be completed before following the algorithm tutorials.

### Folder Structure

We define three main folders that must be prepared apriori:

* `/input/` contains one of the [PI-CAI datasets](https://pi-cai.grand-challenge.org/DATA/). This can be the Public Training and Development Dataset, the Private Training Dataset, the Hidden Validation and Tuning Cohort, or the Hidden Testing Cohort.
  * `/input/images/` contains the imaging files. For the Public Training and Development Dataset, these can be retrieved [here](https://zenodo.org/record/6624726).
  * `/input/labels/` contains the annotations. For the Public Training and Development Dataset, these can be retrieved [here](https://github.com/DIAGNijmegen/picai_labels).
* `/workdir/` stores intermediate results, such as preprocessed images and annotations.
  * `/workdir/results/[model name]/` stores model checkpoints/weights during training (enables the ability to pause/resume training).
* `/output/` stores training output, such as trained model weights and preprocessing plan.

### Data Preparation

Unless specified otherwise, this tutorial assumes that the [PI-CAI: Public Training and Development Dataset](https://pi-cai.grand-challenge.org/DATA/) will be downloaded and unpacked. Before downloading the dataset, read its [documentation](https://zenodo.org/record/6624726) and [dedicated forum post](https://grand-challenge.org/forums/forum/pi-cai-607/topic/public-training-and-development-dataset-updates-and-fixes-631/) (for all updates/fixes, if any). To download and unpack the dataset, run the following commands:

```shell
# download all folds
curl -C - "https://zenodo.org/record/6624726/files/picai_public_images_fold0.zip?download=1" --output picai_public_images_fold0.zip
curl -C - "https://zenodo.org/record/6624726/files/picai_public_images_fold1.zip?download=1" --output picai_public_images_fold1.zip
curl -C - "https://zenodo.org/record/6624726/files/picai_public_images_fold2.zip?download=1" --output picai_public_images_fold2.zip
curl -C - "https://zenodo.org/record/6624726/files/picai_public_images_fold3.zip?download=1" --output picai_public_images_fold3.zip
curl -C - "https://zenodo.org/record/6624726/files/picai_public_images_fold4.zip?download=1" --output picai_public_images_fold4.zip

# unzip all folds
unzip picai_public_images_fold0.zip -d /input/images/
unzip picai_public_images_fold1.zip -d /input/images/
unzip picai_public_images_fold2.zip -d /input/images/
unzip picai_public_images_fold3.zip -d /input/images/
unzip picai_public_images_fold4.zip -d /input/images/
```

In case `unzip` is not installed, you can use Docker to unzip the files:

```shell
docker run --cpus=2 --memory=8gb --rm -v /path/to/input:/input yuanyuan29/z-ssmnet:latest unzip /input/picai_public_images_fold0.zip -d /input/images/
docker run --cpus=2 --memory=8gb --rm -v /path/to/input:/input yuanyuan29/z-ssmnet:latest unzip /input/picai_public_images_fold1.zip -d /input/images/
docker run --cpus=2 --memory=8gb --rm -v /path/to/input:/input yuanyuan29/z-ssmnet:latest unzip /input/picai_public_images_fold2.zip -d /input/images/
docker run --cpus=2 --memory=8gb --rm -v /path/to/input:/input yuanyuan29/z-ssmnet:latest unzip /input/picai_public_images_fold3.zip -d /input/images/
docker run --cpus=2 --memory=8gb --rm -v /path/to/input:/input yuanyuan29/z-ssmnet:latest unzip /input/picai_public_images_fold4.zip -d /input/images/
```

Please follow the [instructions here](https://github.com/yuanyuan29/Z-SSMNet/blob/master/z-nnmnet.md#z-nnmnet---docker-setup) to set up the Docker container.

Also, collect the training annotations via the following command:

```shell
git clone https://github.com/DIAGNijmegen/picai_labels /input/labels/
```

### Cross-Validation Splits

We use the PI-CAI challenge organizers prepared 5-fold cross-validation splits of all 1500 cases in the [PI-CAI: Public Training and Development Dataset](https://pi-cai.grand-challenge.org/DATA/). There is no patient overlap between training/validation splits. You can load these splits as follows:

```python
from z_ssmnet.splits.picai import train_splits, valid_splits

for fold, ds_config in train_splits.items():
    print(f"Training fold {fold} has cases: {ds_config['subject_list']}")

for fold, ds_config in valid_splits.items():
    print(f"Validation fold {fold} has cases: {ds_config['subject_list']}")
```

Additionally, the organizers prepared 5-fold cross-validation splits of all cases with an [expert-derived csPCa annotation](https://github.com/DIAGNijmegen/picai_labels/tree/main/csPCa_lesion_delineations/human_expert). These splits are subsets of the splits above. You can load these splits as follows:

```python
from z_ssmnet.splits.picai_nnunet import train_splits, valid_splits
```

When using `picai_eval` from the command line, we recommend saving the splits to disk. Then, you can pass these to `picai_eval` to ensure all cases were found. You can export the labelled cross-validation splits using:

```shell
python -m z_ssmnet.splits.picai_nnunet --output "/workdir/splits/picai_nnunet"
```

### Data Preprocessing

We follow the [`nnU-Net Raw Data Archive`](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md) format to prepare our dataset for usage. For this, you can use the [`picai_prep`](https://github.com/DIAGNijmegen/picai_prep) module. Note, the [`picai_prep`](https://github.com/DIAGNijmegen/picai_prep) module should be automatically installed when installing the `Z-SSMNet` module, and is installed within the `z-ssmnet` Docker container as well.

To convert the dataset in `/input/` into the [`nnU-Net Raw Data Archive`](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md) format, and store it in `/workdir/nnUNet_raw_data`, please follow the instructions [provided here](https://github.com/DIAGNijmegen/picai_prep#mha-archive--nnu-net-raw-data-archive), or set your target paths in [`prepare_data_semi_supervised.py`](https://github.com/yuanyuan29/Z-SSMNet/blob/master/src/z_ssmnet/prepare_data_semi_supervised.py) and execute it:

```shell
python src/z_ssmnet/prepare_data_semi_supervised.py
```

To adapt/modify the preprocessing pipeline or its default specifications, please make changes to the [`prepare_data_semi_supervised.py`](https://github.com/yuanyuan29/Z-SSMNet/blob/master/src/z_ssmnet/prepare_data_semi_supervised.py) script accordingly.

Alternatively, you can use Docker to run the Python script:

```shell
docker run --cpus=2 --memory=16gb --rm \
    -v /path/to/input/:/input/ \
    -v /path/to/workdir/:/workdir/ \
    -v /path/to/Z-SSMNet:/scripts/Z-SSMNet/ \
    yuanyuan29/z-ssmnet:latest python3 /scripts/Z-SSMNet/src/z_ssmnet/prepare_data_semi_supervised.py
```

If you want to train the supervised model (only using the data with manual labels), prepare the dataset using [`prepare_data.py`](https://github.com/yuanyuan29/Z-SSMNet/blob/master/src/z_ssmnet/prepare_data.py) and replace `Task2302_z-nnmnet` with `Task2301_z-nnmnet` in the following commands.

## Model Training

The implementation of the model consists of three main parts:

* [Zonal Segmentation](https://github.com/yuanyuan29/Z-SSMNet/tree/master#zonal-segmentation)
* [SSL Pre-training](https://github.com/yuanyuan29/Z-SSMNet/tree/master#ssl-pre-training)
* [Z-nnMNet](https://github.com/yuanyuan29/Z-SSMNet/tree/master#z-nnmnet)

### Zonal Segmentation

Prostate area is consist of peripharal zone (PZ), transition zone (TZ), central zone (CZ) and anterior fibromuscular stroma (AFS). Prostate cancer (PCa) lesions located in different zones have different characteristics. Moreover, approximately 70%-75% of PCa originate in the PZ and 20%-30% in the TZ [[1]](https://www.sciencedirect.com/science/article/pii/S0302283815008489). In this work, we trained a standard `3D nnU-Net` [[2]](https://www.nature.com/articles/s41592-020-01008-z) with external public datasets to generate binary prostate zonal anatomy masks (peripheral and rest (TZ, CZ, AFS) of the gland) as additional input information to guide the network to learn region-specific knowledge useful for clinically significant PCa (csPCa) detection and diagnosis.

[→ Read the full documentation here](https://github.com/yuanyuan29/Z-SSMNet/blob/master/zonal_segmentation.md).

### SSL Pre-training

SSL is a general learning framework that relies on surrogate (pretext) tasks that can be formulated using only unsupervised data. A pretext task is designed in a way that solving it requires learning of valuable image representations for the downstream (main) task, which contributes to improve the generalization ability and performance of the model. We introduced image restoration as the pretext task and pre-trained our zonal-aware mesh network in a self-supervised manner.

[→ Read the full documentation here](https://github.com/yuanyuan29/Z-SSMNet/blob/master/ssl.md).

### Z-nnMNet

Considering the heterogeneous between data from multi-centres and multi-vendors, we integrated the zonal-aware mesh network into the famous nnU-Net framework which provides a performant framework for medical image segmentation to form the `Z-nnMNet` that can pre-process the data adaptively. For large datasets with labels, the model can be trained from scratch. If the dataset is small or some labels of the data are noisy, fine-tuning based on the SSL pre-trained model can help achieve better performance.

[→ Read the full documentation here](https://github.com/yuanyuan29/Z-SSMNet/blob/master/z-nnmnet.md).

## References

[[1]](https://www.sciencedirect.com/science/article/pii/S0302283815008489) J. C. Weinreb, J. O. Barentsz, P. L. Choyke, F. Cornud, M. A. Haider, K. J. Macura, D. Margolis, M. D. Schnall, F. Shtern, C. M. Tempany, H. C. Thoeny, and S. Verma, “PI-RADS Prostate Imaging - Reporting and Data System: 2015, Version 2,” European Urology, vol. 69, no. 1, pp. 16-40, Jan, 2016.

[[2]](https://www.nature.com/articles/s41592-020-01008-z) F. Isensee, P. F. Jaeger, S. A. A. Kohl, J. Petersen, and K. H. Maier-Hein, “nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation,” Nature Methods, vol. 18, no. 2, pp. 203-+, Feb, 2021.

[[3]](https://arxiv.org/abs/2205.04846) Z. Dong, Y. He, X. Qi, Y. Chen, H. Shu, J.-L. Coatrieux, G. Yang, and S. Li, “MNet: Rethinking 2D/3D Networks for Anisotropic Medical Image Segmentation,” arXiv:2205.04846, 2022.

[[4]](https://www.sciencedirect.com/science/article/pii/S1361841520302048) Z. Zhou, V. Sodha, J. Pang, M. B. Gotway, and J. Liang, “Models Genesis,” Med Image Anal, vol. 67, pp. 101840, Jan, 2021.

[[5]](https://zenodo.org/record/6667655) A. Saha, J. J. Twilt, J. S. Bosma, B. van Ginneken, D. Yakar, M. Elschot, J. Veltman, J. J. Fütterer, M. de Rooij, H. Huisman, "Artificial Intelligence and Radiologists at Prostate Cancer Detection in MRI: The PI-CAI Challenge (Study Protocol)", DOI: 10.5281/zenodo.6667655
