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
parser.add_argument('-mode')
parser.add_argument('-firstn',type=int)
parser.add_argument('-search')

args = parser.parse_args()

if not os.path.isfile(args.file):
    raise ValueError("Json file does not exists.")

json_file = None
with open(args.file, "r") as f:
    json_file = json.load(f)

if args.firstn is not None:
    print(str(json_file)[:args.firstn])
    exit(1)

#List of dicts
if isinstance(json_file,list):
    print("Number of elements: ",len(json_file))
    if args.search:
        for elem in json_file:
            if str(args.search) in str(elem):
                print(elem)
        exit(1)
        
    
    #unique_imgIds = []
    #for elem in json_file:
    #    if elem['image_id'] not in unique_imgIds:
    #        unique_imgIds.append(elem['image_id'])
    #print("Number of unique image ids: ",len(unique_imgIds))
    print("Sample at position 0: ",json_file[0])
    print("Entries sampled: ", json_file[:2])
elif isinstance(json_file,dict):
    if args.mode == 'annimgs':
        print("Number of images: ",len(json_file['images']))
        print("Sample at position 0: ",json_file['images'][0])
        print("Number of annotations: ",len(json_file['annotations']))
        print("Sample at position 0: ",json_file['annotations'][0])
        print("Sample at position 0-9: ",json_file['annotations'][:10])
    else:
        print("Number of entries: ",len(json_file))
        k = 9
        print("Printing first %d entries:"%k)
        for i,entry in enumerate(json_file.items()):
            print(str(entry)[:70000])
            if i==k-1:
                break





