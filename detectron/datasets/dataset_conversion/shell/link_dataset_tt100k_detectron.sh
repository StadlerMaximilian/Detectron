#!/bin/sh

#####################################
# create sym-links of datasets into
# $Detectron/detectron/datasets/data
# $1 is path to Detectron
# $2 is path to datasets

detect_path="$1"
datset_path="$2"

if [ -d "${dataset_path}/tt100k" ]; then
    echo "Linking tt100k dataset....";
    
    if [ ! -d "${detect_path}/detectron/datasets/data/tt100k"]; then
        mkdir -p "${detect_path}/detectron/datasets/data/tt100k";
    fi
    
    ln -s "${dataset_path}/tt100k/data/test" "${detect_path}/detectron/datasets/data/tt100k/test";
    ln -s "${dataset_path}/tt100k/data/train" "${detect_path}/detectron/datasets/data/tt100k/train";
    ln -s "${dataset_path}/tt100k/data/other" "${detect_path}/detectron/datasets/data/tt100k/other";
    ln -s "${dataset_path}/tt100k/data/JsonAnnotations" "${detect_path}/detectron/datasets/data/tt100k/annotations";
fi

