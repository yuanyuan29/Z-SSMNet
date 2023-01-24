#!/usr/bin/env bash

./build.sh

docker save z-ssmnet | gzip -c > z-ssmnet.tar.gz
