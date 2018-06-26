import json
from pycocotools.coco import COCO
from PIL import Image
import datetime
import argparse
import sys
import os
from conversion_base import CocoConversion


class CocoKittiConversion(CocoConversion):

    def create_json_annos(self):
        pass