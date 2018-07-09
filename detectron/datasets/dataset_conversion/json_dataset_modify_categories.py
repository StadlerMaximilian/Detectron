import json
import sys
import os
import argparse

"""
Utilites to remove categories that should be ignored during testing
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description='Remove categories that sould be ignored during testing'
    )
    parser.add_argument(
        '--json',
        dest='json_file',
        type=str,
        help="Path to your json test-set.",
        default=""
    )

    parser.add_argument(
        '--remove',
        dest='remove_txt',
        type=str,
        help='Path to a .txt-file where you list your categories to be removed.',
        default=""
    )

    parser.add_argument(
        '--keep',
        dest='keep_txt',
        type=str,
        help='path to a.txt-file where you list your categories to be kept.',
        default=''
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.json_file):
        raise ValueError("FILE {} does not exist!!!".format(args.json_file))
    json_file = json.loads(open(args.json_file).read())
    json_file_name = args.json_file.split('.')[0]

    categories = json_file['categories']
    cat_names = [x['name'] for x in categories]
    info = json_file['info']
    images = json_file['images']
    annotations = json_file['annotations']
    licenses = json_file['licenses']

    if args.remove_txt != "":
        if not os.path.exists(args.remove_txt):
            raise ValueError("FILE {} does not exist!!!".format(args.remove_txt))
        remove_cats = open(args.remove_txt).read().split(',')
        new_cat_names = [str(x) for x in cat_names if x not in remove_cats]

    elif args.keep_txt != "":
        if not os.path.exists(args.keep_txt):
            raise ValueError("FILE {} does not exist!!!".format(args.keep_txt))
        keep_cats = open(args.keep_txt).read().split(',')
        new_cat_names = [str(x) for x in keep_cats if x in cat_names]
    else:
        raise ValueError("You have to specify either remove.txt or keep.txt!")

    new_categories = [x for x in categories if x['name'] in new_cat_names]
    new_categories_ids = [x['id'] for x in new_categories]
    new_annotations =[x for x in annotations if x['category_id'] in new_categories_ids]

    new_dataset_dict = {"info": info,
                        "images": images,
                        "annotations": new_annotations,
                        "type": "instances",
                        "licenses": licenses,
                        "categories": new_categories
                        }

    with open(json_file_name + '_ignore.json', 'w') as fp:
        json.dump(new_dataset_dict, fp)


if __name__ == '__main__':
    main()

