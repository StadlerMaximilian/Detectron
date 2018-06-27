#!/bin/sh

#####################################
# create sym-links of datasets into
# $Detectron/detectron/datasets/data
# $1 is path to Detectron
# $2 is path to datasets

detect_path="$1"
datset_path="$2"

if [ -d "${dataset_path}/coco" ]; then
    echo "Linking coco dataset....";
    
    if [ ! -d "${detect_path}/detectron/datasets/data/coco"]; then
        mkdir -p "${detect_path}/detectron/datasets/data/coco";
    fi
    
    ln -s "${dataset_path}/coco/coco_test2014" "${detect_path}/detectron/datasets/data/coco/";
    ln -s "${dataset_path}/coco/coco_train2014" "${detect_path}/detectron/datasets/data/coco/";
    ln -s "${dataset_path}/coco/coco_val2014" "${detect_path}/detectron/datasets/data/coco/";
    ln -s "${dataset_path}/coco/annotations" "${detect_path}/detectron/datasets/data/coco/annotations";
fi

