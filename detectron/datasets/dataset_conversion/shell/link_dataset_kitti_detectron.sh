#!/bin/sh

#####################################
# create sym-links of datasets into
# $Detectron/detectron/datasets/data
# $1 is path to Detectron
# $2 is path to datasets

detect_path="$1"
datset_path="$2"

if [ -d "${dataset_path}/kitti" ]; then
    echo "Linking kitti dataset....";
    if [ ! -d "${detect_path}/detectron/datasets/data/kitti"]; then
        mkdir -p "${detect_path}/detectron/datasets/data/kitti";
    fi
    
    ln -s "${dataset_path}/kitti/testing" "${detect_path}/detectron/datasets/data/kitti/testing";
    ln -s "${dataset_path}/kitti/training" "${detect_path}/detectron/datasets/data/kitti/training";
    ln -s "${dataset_path}/kitti/JsonAnnotations" "${detect_path}/detectron/datasets/data/kitti/annotations";
fi

