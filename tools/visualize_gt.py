from pycocotools.coco import COCO
import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Display ground-truth images with bboxes."
    )

    parser.add_argument(
        '--json_file',
        dest='json_file',
        type=str,
        help='Include here the path of the wanted annotations file.',
        default=''
    )

    parser.add_argument(
        '--img_id',
        dest='img_id',
        type=int,
        help='Include here desired image id.',
        default=-1
    )

    parser.add_argument(
        '--img_dir',
        dest='img_dir',
        type=str,
        help='Include here the path to the curresponding image folder.',
        default=''
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.json_file=='':
        raise ValueError('No json_file specified.')
    if args.img_id==-1:
        raise ValueError('No Img_id specified')
    if args.img_dir=='':
        raise ValueError('No img_dir specified')

    if not os.path.exists(args.json_file):
        raise ValueError('Specified json_file does not exist')


    coco = COCO(args.json_file)
    img = coco.loadImgs(args.img_id)
    img = img[0]
    if os.path.exists(args.img_dir + '/' + img['file_name']):
        img_path = args.img_dir + '/' + img['file_name']
    else:
        raise ValueError('Desired image does not exist')

    fig = plt.figure()
    I = mpimg.imread(img_path)
    plt.axis('off')
    plt.imshow(I)
    annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)
    plt.show()


if __name__ == '__main__':
    main()