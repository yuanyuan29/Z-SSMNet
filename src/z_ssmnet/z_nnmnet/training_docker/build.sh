#!/usr/bin/env bash

# read gittoken from file?
if [ -f "gittoken" ];
then
    gittoken=`cat gittoken`
    echo "Using gittoken from file"
else
    echo "Enter your git token (or save to gittoken file): "
    read gittoken
fi

read -p "Enter the branch of yuanyuan29/Z-SSMNet you want to build [master]: " branch
branch=${branch:-master}

echo "Building yuanyuan29/Z-SSMNet docker image from branch $branch"

docker build . \
    --no-cache --build-arg gittoken=$gittoken --build-arg branch=$branch \
    --tag yuanyuan29/z-ssmnet:1.0.0-customized-v1.0 \
    --tag yuanyuan29/z-ssmnet:latest
