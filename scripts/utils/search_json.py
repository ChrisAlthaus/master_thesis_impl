import argparse
import os

from pycocotools.coco import COCO
import datetime
import json
import random
import time

import itertools
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('-file', help='Path to a json file.')
parser.add_argument('-field')
parser.add_argument('-search')



args = parser.parse_args()

if not os.path.isfile(args.file):
    raise ValueError("Json file does not exists.")

json_file = None
with open(args.file, "r") as f:
    json_file = json.load(f)

if isinstance(json_file,list):
    for elem in json_file:
        if str(elem[str(args.field)]).find(args.search) != -1:
            print(elem)
            
elif isinstance(json_file,dict):
    image_ids = []
    for elem in json_file['images']:
        if elem[str(args.field)].find(args.search) != -1:
            print(elem)
            image_ids.append(str(elem['id']))

   
    for elem in json_file['annotations']:
        for id in image_ids:
            if str(elem['image_id']).find(id) != -1:
                print(elem)


