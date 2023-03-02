#!/usr/bin/env bash

./build.sh

docker save z_ssmnet_prostate_zonal_segmentation_processor:latest | gzip -c > z_ssmnet_prostate_zonal_segmentation_processor.tar.gz

