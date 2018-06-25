import json
from pycocotools.coco import COCO
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


class CocoConversion(object):

    def __init__(self, data_dir):
        if not os.path.exists(data_dir):
            raise ValueError("PATH {} does not exist!!!".format(data_dir))
        self.data_dir = data_dir
        self.info = None
        self.licenses = []
        self.categories = []
        self.train_file = None
        self.val_file = None
        self.test_file = None

    @staticmethod
    def category_to_index(category, categories):
        category_id = -1
        for type in categories:
            if type['name'] == category:
                category_id = type['id']
        return category_id

    @staticmethod
    def create_coco_category(id, type, supercategory):
        category = {"id": id,
                    "name": type,
                    "supercategory": supercategory}
        return category

    def append_coco_category(self, id, type, supercategory):
        self.categories.append(self.create_coco_category(id, type, supercategory))

    @staticmethod
    def create_coco_image(file_name, height, width, id):
        image = {"file_name" : file_name,
                 "height": height,
                 "width" : width,
                 "id" : id}
        return image

    def append_coco_image(self, file_name, height, width, id, images):
        images.append(self.create_coco_image(file_name, height, width, id))

    @staticmethod
    def create_coco_annotation(id, image_id, category_id, area, bbox, iscrowd):
        annotation = {"id": id,
                      "image_id": image_id,
                      "category_id": category_id,
                      "area": area,
                      "bbox": bbox,
                      "iscrowd": iscrowd}
        return annotation

    def append_coco_annotation(self, id, image_id, category_id, area, bbox, iscrowd, annotations):
        annotations.append(self.create_coco_annotation(id, image_id, category_id,
                                                       area, bbox, iscrowd))

    def create_coco_info(self, description="", url="", version=0, year=0, contributor="",
                         date_created=datetime.datetime.utcnow().isoformat(' ')):
        info = {
            "description": description,
            "url": url,
            "version": version,
            "year": year,
            "contributor": contributor,
            "date_created": date_created
        }
        self.info = info
        return info

    @staticmethod
    def create_coco_license(id, name, url):
        license ={
                "id": id,
                "name": name,
                "url": url
            }

        return license

    def append_coco_license(self, id=0, name="", url=""):
        self.licenses.append(self.create_coco_license(id, name, url))

    def create_coco_dataset_dict(self, images, annotations):
        dataset_dict = {"info": self.info,
                   "images": images,
                   "annotations": annotations,
                   "licenses": self.licenses,
                   "categories": self.categories
                   }

        return dataset_dict

    def create_json_annos(self):
        """
        overriden in subclass
        subclass must implement this function
        this function should implement the conversion to ms-coco-style-annotations
        and output the according train.json, test.json and val.json file

        the file uses the data_dir specified in the constructor assuming the typical structure of the dataset
        in the original style

        the function should also specify the path to
            the train.json in self.train_file
            the test.json in self.test_file
            the val.json in self.val_file
        """
        raise NotImplementedError("Please Implement this method")

    def check_json_annos(self):
        print("\n")
        print("checking annotation ...")

        files = [self.test_file, self.train_file, self.val_file]

        for file in files:
            print("... checking {}\n".format(file))
            if file is None:
                raise ValueError("FILE {} not found".format(file))
            else:
                # initialize COCO api for instance annotations
                try:
                    coco = COCO(file)
                except:
                    print("ERROR occured: conversion failed !!!")
                    return

                try:
                    cats = coco.loadCats(coco.getCatIds())
                    nms = [cat['name'] for cat in cats]
                    print('dataset categories: \n{}\n'.format(' '.join(nms)))

                    nms = set([cat['supercategory'] for cat in cats])
                    print('dataset supercategories: {}\n'.format(' '.join(nms)))
                except:
                    print("ERROR occured: conversion failed !!!")
                    raise

                try:
                    imgs = coco.loadImgs(coco.getImgIds())
                    img_ids = [img['id'] for img in imgs]
                    annIds = coco.getAnnIds(imgIds=img_ids, iscrowd=None)
                    anns = coco.loadAnns(annIds)
                    print("{} contains {} images and {} annotations".format(file, len(img_ids), len(anns)))
                except:
                    print("ERROR occured: conversion failed !!!")
                    raise

                print("sub-check completed\n")
        print("created following json files")
        print("...test:" + files[0])
        print("...train:" + files[1])
        print("...val:" + files[2])
        print("conversion completed")


class CocoTt100kConversion(CocoConversion):

    def create_json_annos(self):
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

        print("Converting TT100k dataset .... ")

        file_dir = self.data_dir + "/annotations.json"
        if not os.path.exists(file_dir):
            raise ValueError("FILE {} does not exist!!!".format(file_dir))
        annos = json.loads(open(file_dir).read())

        self.info = self.create_coco_info()
        self.append_coco_license() #empty license

        types = annos["types"]
        for type_idx, type in enumerate(types, 1):
            self.append_coco_category(type_idx, type, "none")

        ann_id_counter = 1
        for set in ["train", "test", "other"]:
            img_count = 0
            images = []
            annotations = []

            # create json for specific set
            ids_path = self.data_dir + "/" + set + "/ids.txt"
            if not os.path.exists(ids_path):
                raise ValueError("FILE {} does not exist!!!".format(ids_path))
            ids = open(ids_path).read().splitlines()
            for imgid in ids:
                img = annos["imgs"][imgid]
                file_name = img['path'].split('/')[1]
                img_path = self.data_dir + "/" + set + "/" + file_name
                if not os.path.exists(img_path):
                    print("IMAGE {} does not exist, continuing ... ".format(file_name))
                    continue
                img_file = Image.open(img_path)
                img_count += 1
                width, height = img_file.size
                image_id = int(file_name.split('.')[0])
                self.append_coco_image(file_name, height, width, image_id, images)

                for obj in img['objects']:
                    category = obj["category"]
                    bbox_tt100k = obj["bbox"]

                    # tt100k: format [xmin, ymin, ymax, xmax] as dict

                    # coco: box coordinates are measured from the top left image corner and are 0-indexed
                    # format [x,y,width, height]

                    id = ann_id_counter
                    ann_id_counter += 1
                    category_id = self.category_to_index(category, self.categories)
                    x = bbox_tt100k['xmin']  # left corner
                    y = bbox_tt100k['ymax']  # top left corner
                    width = bbox_tt100k['xmax'] - bbox_tt100k['xmin']
                    height = bbox_tt100k['ymax'] - bbox_tt100k['ymin']
                    area = width * height
                    bbox = [x, y, width, height]
                    iscrowd = 0
                    self.append_coco_annotation(id, image_id, category_id, area,
                                                bbox, iscrowd, annotations)

            print("..." + set + ": converted {} images".format(img_count))

            dataset_dict = self.create_coco_dataset_dict(images, annotations)
            with open(self.data_dir + '/tt100k_' + set + '.json', 'w') as fp:
                json.dump(dataset_dict, fp)
                if set == "other":
                    self.val_file = self.data_dir + '/tt100k_' + set + '.json'
                elif set == "train":
                    self.train_file = self.data_dir + '/tt100k_' + set + '.json'
                elif set == "test":
                    self.test_file = self.data_dir + '/tt100k_' + set + '.json'


class CocoCaltechPedestrianConversion(CocoConversion):

    def create_json_annos(self):
        pass


class CocoKittiConversion(CocoConversion):

    def create_json_annos(self):
        pass


class CocoVkittiConversion(CocoConversion):

    def create_json_annos(self):
        pass


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert a dataset to ms-coco-style format'
    )
    parser.add_argument(
        '--data_dir',
        dest='data_dir',
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
        dataset = CocoTt100kConversion(args.data_dir)
    elif args.data_type == 'caltech_pedestrian':
        dataset = CocoCaltechPedestrianConversion(args.data_dir)
    elif args.data_type == 'kitti':
        dataset = CocoKittiConversion(args.data_dir)
    elif args.data_type == 'vkitti':
        dataset = CocoVkittiConversion(args.data_dir)
    else:
        raise ValueError("Specified type of dataset is not supported.")

    dataset.create_json_annos()
    dataset.check_json_annos()


if __name__ == '__main__':
    main()

