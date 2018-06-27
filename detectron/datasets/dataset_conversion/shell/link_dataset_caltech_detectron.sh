#!/bin/sh

#####################################
# create sym-links of datasets into
# $Detectron/detectron/datasets/data
# $1 is path to Detectron
# $2 is path to datasets

detect_path="$1"
datset_path="$2"

if [ -d "${dataset_path}/caltech_pedestrian" ]; then
    echo "Linking caltech_pedestrian dataset....";
    
    if [ ! -d "${detect_path}/detectron/datasets/data/caltech_pedestrian"]; then
        mkdir -p "${detect_path}/detectron/datasets/data/caltech_pedestrian";
    fi
    
    ln -s "${dataset_path}/caltech_pedestrian/train" "${detect_path}/detectron/datasets/data/caltech_pedestrian/train";
    ln -s "${dataset_path}/caltech_pedestrian/test" "${detect_path}/detectron/datasets/data/caltech_pedestrian/test";
    ln -s "${dataset_path}/caltech_pedestrian/JsonAnnotations" "${detect_path}/detectron/datasets/data/caltech_pedestrian/annotations";
fi

