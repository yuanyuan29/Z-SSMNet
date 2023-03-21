# Adapted from https://github.com/DIAGNijmegen/diag-nnunet/blob/master/Dockerfile

FROM nvcr.io/nvidia/pytorch:20.12-py3
ARG gittoken
ARG branch=master

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

# Configuration
RUN echo "PYTHONUNBUFFERED=1" >> /etc/environment && \
    echo "OMP_NUM_THREADS=1" >> /etc/environment
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Install a few dependencies that are not automatically installed, plus nnU-Net
RUN apt-get update && \
    apt-get install -y --no-install-recommends graphviz && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN git config --global advice.detachedHead false && \
    git clone https://github.com/DIAGNijmegen/nnUNet.git /home/user/nnunet && \
    git -C /home/user/nnunet checkout 1.7.0-3 && \
    chown -R user /home/user/nnunet && \
    pip3 install \
        -e /home/user/nnunet \
        graphviz \
        git+https://github.com/nanohanno/hiddenlayer.git@bugfix/get_trace_graph#egg=hiddenlayer \
        carbontracker && \
    rm -rf ~/.cache/pip

# Install MSK-Tiger IO subset
COPY io.py /tmp/io.py
RUN SITE_PKG=`pip3 show nnunet | grep "Location:" | awk '{print $2}'` && \
    mv /tmp/io.py "$SITE_PKG/nnunet/utilities/io.py"


# Install PI-CAI repositories
RUN pip3 install \
    picai_eval==1.4.4 \
    picai_prep==2.1.3 \
    picai_baseline==0.8.2 \
    mlxtend==0.19.0

# Install SSL required modules
RUN pip3 install tensorboard 

# copy custom files to nnUNet folder
COPY nnUNetTrainerV2_focalLoss.py /tmp/nnUNetTrainerV2_focalLoss.py
RUN SITE_PKG=`pip3 show nnunet | grep "Location:" | awk '{print $2}'` && \
    mv /tmp/nnUNetTrainerV2_focalLoss.py "$SITE_PKG/nnunet/training/network_training/nnUNet_variants/loss_function/nnUNetTrainerV2_focalLoss.py"

COPY MNet.py /tmp/MNet.py
RUN SITE_PKG=`pip3 show nnunet | grep "Location:" | awk '{print $2}'` && \
    mv /tmp/MNet.py "$SITE_PKG/nnunet/network_architecture/MNet.py"

COPY MNet_basic_module.py /tmp/MNet_basic_module.py
RUN SITE_PKG=`pip3 show nnunet | grep "Location:" | awk '{print $2}'` && \
    mv /tmp/MNet_basic_module.py "$SITE_PKG/nnunet/network_architecture/MNet_basic_module.py"

COPY MNet_myTrainer_zonal.py /tmp/MNet_myTrainer_Focal_zonal.py
RUN SITE_PKG=`pip3 show nnunet | grep "Location:" | awk '{print $2}'` && \
    mv /tmp/MNet_myTrainer_Focal_zonal.py "$SITE_PKG/nnunet/training/network_training/MNet_myTrainer_zonal.py"

COPY dataset_loading.py /tmp/dataset_loading.py
RUN SITE_PKG=`pip3 show nnunet | grep "Location:" | awk '{print $2}'` && \
    mv /tmp/dataset_loading.py "$SITE_PKG/nnunet/training/dataloading/dataset_loading.py"

COPY nnUNetTrainer.py /tmp/nnUNetTrainer.py
RUN SITE_PKG=`pip3 show nnunet | grep "Location:" | awk '{print $2}'` && \
    mv /tmp/nnUNetTrainer.py "$SITE_PKG/nnunet/training/network_training/nnUNetTrainer.py"

COPY predict.py /tmp/predict.py
RUN SITE_PKG=`pip3 show nnunet | grep "Location:" | awk '{print $2}'` && \
    mv /tmp/predict.py "$SITE_PKG/nnunet/inference/predict.py"

COPY run_training.py /tmp/run_training.py
RUN SITE_PKG=`pip3 show nnunet | grep "Location:" | awk '{print $2}'` && \
    mv /tmp/run_training.py "$SITE_PKG/nnunet/run/run_training.py"

# Install Z-SSMNet module
WORKDIR /opt/algorithm
RUN git clone https://${gittoken}@github.com/yuanyuan29/Z-SSMNet --branch ${branch} --single-branch
RUN pip3 install /opt/algorithm/Z-SSMNet

# Configure entrypoint
COPY --chown=user nnunet_wrapper.py /home/user/nnunet_wrapper.py
RUN chmod +x /home/user/nnunet_wrapper.py && \
    ln -s /home/user/nnunet_wrapper.py /usr/local/bin/nnunet

RUN chmod +x /opt/algorithm/Z-SSMNet/src/z_ssmnet/training/train_z_ssmnet.py && \
    ln -s /opt/algorithm/Z-SSMNet/src/z_ssmnet/training/train_z_ssmnet.py /usr/local/bin/train
