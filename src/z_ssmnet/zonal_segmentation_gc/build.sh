#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build "$SCRIPTPATH" \
    -t yuanyuan29/z_ssmnet_prostate_segmentation_processor:latest \
    -t yuanyuan29/z_ssmnet_prostate_segmentation_processor:v1.0
