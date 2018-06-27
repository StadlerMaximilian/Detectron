#!/bin/sh

#####################################
# create sym-links of datasets into
# $Detectron/detectron/datasets/data
# $1 is path to Detectron
# $2 is path to datasets

detect_path="$1"
datset_path="$2"

if [ -d "${dataset_path}/vkitti" ]; then
    echo "Linking vkitti dataset....";
    if [ ! -d "${detect_path}/detectron/datasets/data/vkitti"]; then
        mkdir -p "${detect_path}/detectron/datasets/data/vkitti";
    fi
    
    ln -s "${dataset_path}/vkitti/Images" "${detect_path}/detectron/datasets/data/kitti/Images";
    ln -s "${dataset_path}/vkitti/JsonAnnotations" "${detect_path}/detectron/datasets/data/vkitti/annotations";
fi

