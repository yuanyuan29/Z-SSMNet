#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build "$SCRIPTPATH" \
    -t z_ssmnet_prostate_zonal_segmentation_processor:latest \
    -t z_ssmnet_prostate_zonal_segmentation_processor:v1.0
