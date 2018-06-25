import json
from PIL import Image
import datetime
import argparse
import sys
import os

"""
ms coco style
{
    "info": info, optional
    "images": [image]
    "annotations": [annotation]
    "licenses": licence, optinal
    "categories": categories, for object detection
}
    
    info = {
        "year": int
        "version": str
        "description": str
        "contributor": str
        "url": str
        "date_created": datetime
    }
    
    licence = {
        "id": id
        "name": str
        "url": str
    }
    
    image = {
        "file_name": str
        "height": int
        "width": int
        "id": int #use number from file-name
        "licencce": int, optional
        "flickr_url": str, optinal
        "coco_url": str, optional
        "date_captured": datetime
    }
    
    annotation = {
        "id": int,
        "image_id" : int,
        "category_id" : int,
        "segmentation": RLE or [polygon],
        "area" : float,
        "bbox" : [x,y,width,height],
        "iscrowd" : 0 or 1,
        }
    if iscrowd == 0: polyongs used (single object, several polyongs may be needed due to occlusions)
    if iscrowd == 1: RLE coding of binary mask (for larger objects, like groups of people)
        
    categories = [{
        "id" : int,
        "name" : str,
        "supercategory" : str,
        }]
    
"""




def create_json_annos_tt100k(datadir):
    """
    TT100K JSON STYLE
    { "imgs": imgs,
      "types": types
    }

    imgs = {
        "id":
    }

    image = {
        "path": str
        "objects": [object]
    }

    object = {
        "category": str
        "bbox": {"xmin", "ymin", "ymax", "xmax"}
        "ellipse_org": ..., optional
        "ellipse": ..., optinal
        "polygon". ..., optional
    }

    types = [type] where type: str
    """

    categories = []
    images = []
    annotations = []

    filedir = datadir + "/annotations.json"
    annos = json.loads(open(filedir).read())

    info = create_coco_info()
    licenses = [create_coco_licence()]

    types = annos["types"]
    for type_idx, type in enumerate(types, 1):
        categories.append(create_coco_category(type_idx, type, "none"))

    ann_id_counter = 1
    for set in ["train", "test", "other"]:
        # create json for specific set

        ids = open(datadir + "/" + set + "/ids.txt").read().splitlines()
        for imgid in ids:
            img = annos["imgs"][imgid]
            file_name = img['path'].split('/')[1]
            img_file = Image.open(datadir + "/" + set + "/" + file_name)
            width, height =  img_file.size
            image_id = int(file_name.split('.')[0])
            images.append(create_coco_image(file_name,
                                            height,
                                            width,
                                            image_id))

            for obj in img['objects']:
                category = obj["category"]
                bbox_tt100k = obj["bbox"]

                #tt100k: format [xmin, ymin, ymax, xmax] as dict

                #coco: box coordinates are measured from the top left image corner and are 0-indexed
                #format [x,y,width, height]

                id = ann_id_counter
                ann_id_counter += 1
                category_id = types[category]
                area = 0.0 #no segmentation support
                segmentation = [[]] #no segmentation support
                x = bbox['xmin'] #left corner
                y = bbox_tt100k['ymax'] #top left corner
                width = bbox_tt100k['xmax'] - bbox_tt100k['xmin']
                height = bbox_tt100k['ymax'] - bbox_tt100k['ymin']
                bbox = [x,y, width, height]
                iscrowd = 0
                annotations.append(create_coco_annotation(id, image_id, category_id,
                                                          segmentation,
                                                          area,
                                                          bbox,
                                                          iscrowd))
        dataset = {"info" : info,
                   "images" : images,
                   "annotations" : annotations,
                   "licenses" : licenses,
                   "categories" : categories
                   }

        with open(datadir + '/tt100k_' + set + '.json', 'w') as fp:
            json.dump(dataset, fp)


def create_coco_category(id, type, supercategory):
    category = {"id": id,
                "name": type,
                "supercategory": supercategory}
    return category


def create_coco_image(file_name, height, width, id):
    image = {"file_name" : file_name,
             "height": height,
             "width" : width,
             "id" : id}
    return image


def create_coco_annotation(id, image_id, category_id, segmentation, area, bbox, iscrowd):
    annotation = {"id": id,
                  "image:id": image_id,
                  "category_id": category_id,
                  "segmentation": segmentation,
                  "area": area,
                  "bbox": bbox,
                  "iscrowd": iscrowd}
    return annotation


def create_coco_info(description="", url="", version=0, year=0, contributor="",
                     date_created=datetime.datetime.utcnow().isoformat(' ')):
    info = {
        "description": description,
        "url": url,
        "version": version,
        "year": year,
        "contributor": contributor,
        "date_created": date_created
    }

    return info


def create_coco_licence(id=0, name="", url=""):
    license ={
            "id": 1,
            "name": name,
            "url": url
        }

    return license

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert a dataset to ms-coco-style format'
    )
    parser.add_argument(
        '--datadir',
        dest='datadir',
        help='Include here the path to the root directory where your dataset is stored',
        default=None,
        type=str
    )

    parser.add_argument(
        '--type',
        dest='dataset_type',
        help='Include here the type of your dataset you want to convert. At the moment, supported datasets are: tt100k',
        default=None,
        type=str
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dataset_type is None or args.dataset_type is None:
        return
    if args.dataset_type == 'tt100k':
        create_json_annos_tt100k(args.datadir)
    else:
        return


if __name__ == '__main__':
    main()
