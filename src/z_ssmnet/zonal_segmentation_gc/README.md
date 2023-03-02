# Prostate Zonal Segmentation in MRI

## Managed By

Yuan Yuan, Biomedical Data Analysis and Visualisation (BDAV) Lab, The University of Sydney, Sydney, Australia

## Contact Information

Yuan Yuan: yyua9990@uni.sydney.edu.au

## Inference

This algorithm is hosted on [Grand-Challenge.com](https://grand-challenge.org/algorithms/bdav-prostate-zonal-segmentation/).

## Summary

This algorithm segments the peripheral zone (PZ) and central gland (CG) (transition zone (TZ) + central zone (CZ) + anterior fibromuscular stroma (AFS)) of prostate in biparametric MRI (bpMRI) (T2W + ADC). Development of this model was geared toward robust prostate zonal segmentation. This algorithm was used to provide prostate zonal segmentations of our [Z-SSMNet](https://arxiv.org/pdf/2212.05808.pdf) model for the [PI-CAI challenge](https://pi-cai.grand-challenge.org/).

## Mechanism

This algorithm is a 3D nnU-Net model. We trained the model with Cross-Entropy + Dice loss. The ratio of training set and test set is 4:1. We trained the model with a total of 394 prostate biparametric MRI (bpMRI) scans paired with a manual prostate zonal segmentation. These scans were sourced from two independent hospitals: 204 cases from [ProstateX](https://prostatex.grand-challenge.org/), 158 cases from [Prostate158](https://prostate158.grand-challenge.org/) and 32 cases from [Medical Segmentation Decathlon (MSD)](http://medicaldecathlon.com/).

We ensured there is no patient overlap between this algorithm's training dataset and the [PI-CAI](https://pi-cai.grand-challenge.org/) [Hidden Validation and Tuning Cohort](https://pi-cai.grand-challenge.org/DATA/) or [Hidden Testing Cohort](https://pi-cai.grand-challenge.org/DATA/).

## Segmentation Performance

Segmentation performance are provided below.

|              | Dice similarity coefficient | Jaccard index |
| ------------ | --------------------------- | ------------- |
| **PZ** | 0.7695                      | 0.6377        |
| **CG** | 0.8659                      | 0.7772        |

## Uses and Directions

* **For research use only** . This algorithm is intended to be used only on biparametric prostate MRI examinations. This algorithm should not be used in different patient demographics.
* **Target population** : This algorithm was trained on patients without prior treatment (e.g. radiotherapy, transurethral resection of the prostate (TURP), transurethral ultrasound ablation (TULSA), cryoablation, etc.), without prior positive biopsies, without artefacts and with reasonably-well aligned sequences.
* **MRI scanner** : This algorithm was trained and evaluated exclusively on prostate bpMRI scans derived from Siemens Healthineers (Skyra/Prisma/Trio/Avanto) MRI scanners with surface coils. It does not account for vendor-neutral properties or domain adaptation, and in turn, is not compatible with scans derived using any other MRI scanner or those using endorectal coils.
* **Sequence alignment and position of the prostate** : While the input images (T2W, ADC) can be of different spatial resolutions, the algorithm assumes that they are co-registered or aligned reasonably well.
* **General use** : This model is intended to be used by radiologists for predicting prostate zonal area in biparametric MRI examinations. The model is not meant to guide or drive clinical care. This model is intended to complement other pieces of patient information in order to determine the appropriate follow-up recommendation.
* **Before using this model** : Test the model retrospectively and prospectively on a cohort that reflects the target population that the model will be used upon to confirm the validity of the model within a local setting.
* **Safety and efficacy evaluation** : To be determined in a clinical validation study.

## Warnings

* **Risks** : Even if used appropriately, clinicians using this model can estimate prostate zonal area incorrectly.
* **Inappropriate Settings** : This model was not trained on MRI examinations of patients with prior treatment (e.g. radiotherapy, transurethral resection of the prostate (TURP), transurethral ultrasound ablation (TULSA), cryoablation, etc.), prior positive biopsies, artefacts or misalignment between sequences. Hence it is susceptible to faulty predictions and unintended behaviour when presented with such cases. Do not use the model in the clinic without further evaluation.
* **Clinical rationale** : The model is not interpretable and does not provide a rationale. Clinical end users are expected to place the model output in context with other clinical information.
* **Inappropriate decision support** : This model may not be accurate outside of the target population. This model is not designed to guide clinical diagnosis and treatment for prostate cancer.
* **Generalizability** : This model was primarily developed with prostate MRI examinations from Radboud University Medical Centre and the Charité University Hospital Berlin. Do not use this model in an external setting without further evaluation.
* **Discontinue use if** : Clinical staff raise concerns about the utility of the model for the intended use case or large, systematic changes occur at the data level that necessitates re-training of the model.
