#!/usr/bin/env bash

./build.sh

docker save yuanyuan29/z_ssmnet_prostate_segmentation_processor:latest | gzip -c > yuanyuan29_z_ssmnet_prostate_segmentation_processor.tar.gz

