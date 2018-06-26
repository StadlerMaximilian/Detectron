import json
from pycocotools.coco import COCO
from PIL import Image
import datetime
import argparse
import sys
import os

from tt100k_conversion import CocoTt100kConversion
from kitti_conversion import CocoKittiConversion
from vkitti_conversion import CocoVkittiConversion
from caltech_pedestrian_conversion import CocoCaltechPedestrianConversion


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert a dataset to ms-coco-style format'
    )
    parser.add_argument(
        '--data_dir',
        dest='data_dir',
        type=str,
        help='Include here the path to the root directory where your dataset is stored',
        default=""
    )

    parser.add_argument(
        '--type',
        dest='dataset_type',
        type=str,
        help='Include here the type of your dataset you want to convert. At the moment, supported datasets are: '
             'tt100k, caltech_pedestrian, kitti, vkitti',
        default=""
    )

    parser.add_argument(
        '--check',
        dest='check_bool',
        type= bool,
        help='Set bool to false for disabling check_json_annos',
        default=False
    )

    parser.add_argument(
        '--plot',
        dest='plot_bool',
        help='Set bool to false for disabling ploting example annotations',
        default=False
    )

    parser.add_argument(
        '--overfit_sample',
        dest='overfit_bool',
        help='Set bool to false for disabling the creation of an overfitting annotation file',
        default=False
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main():
    args = parse_args()
    im_dir = ""
    json_file = ""
    if args.dataset_type is None or args.dataset_type is None:
        return
    if args.dataset_type == 'tt100k':
        dataset = CocoTt100kConversion(args.data_dir)
        im_dir = args.data_dir + '/train'
        json_file = args.data_dir + '/tt100k_train.json'
    elif args.dataset_type == 'caltech_pedestrian':
        dataset = CocoCaltechPedestrianConversion(args.data_dir)
        im_dir = args.data_dir + '/train'
        json_file = args.data_dir + '/caltech_original_train.json'
    elif args.dataset_type == 'kitti':
        dataset = CocoKittiConversion(args.data_dir)
        im_dir = args.data_dir + '/train'
        json_file = args.data_dir + '/kitti_train.json'
    elif args.dataset_type == 'vkitti':
        dataset = CocoVkittiConversion(args.data_dir)
        im_dir = args.data_dir + '/train'
        json_file = args.data_dir + '/vkitti_train.json'
    else:
        raise ValueError("Specified type of dataset is not supported.")

    dataset.create_json_annos()

    if args.check_bool:
        dataset.check_json_annos(args.plot_bool, im_dir, json_file)
    if args.overfit_bool:
        dataset.create_json_overfit()


if __name__ == '__main__':
    main()

