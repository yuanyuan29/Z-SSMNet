FROM joeranbosma/picai_nnunet:1.7.0-customized-v1.4

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

RUN apt-get -y update
# RUN apt-get -y install git

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

RUN mkdir -p /opt/algorithm/results/ \
    && chown algorithm:algorithm /opt/algorithm/results/
COPY --chown=algorithm:algorithm results/ /opt/algorithm/results/

# Install algorithm requirements
COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user -r requirements.txt

# Copy the processor to the algorithm container folder
COPY --chown=algorithm:algorithm process.py /opt/algorithm/

ENTRYPOINT python -m process $0 $@

## ALGORITHM LABELS ##

# These labels are required
LABEL nl.diagnijmegen.rse.algorithm.name=z_ssmnet_prostate_zonal_segmentation_processor
