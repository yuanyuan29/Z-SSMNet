FROM yuanyuan29/z-ssmnet:1.0.0-customized-v1.0

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

# Install algorithm requirements
COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user -r requirements.txt

# Copy the processor to the algorithm container folder
COPY --chown=algorithm:algorithm process.py /opt/algorithm/

ENTRYPOINT python -m process $0 $@

## ALGORITHM LABELS ##

# These labels are required
LABEL nl.diagnijmegen.rse.algorithm.name=yuanyuan29_z_ssmnet_prostate_segmentation_processor
