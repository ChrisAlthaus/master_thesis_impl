import argparse
import os

import datetime
import json
import random
import time

import itertools
import numpy as np

from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('-file', help='Path to a json file.')
parser.add_argument('-mode')
parser.add_argument('-firstn',type=int)
parser.add_argument('-search')
parser.add_argument('-custom',action='store_true')

args = parser.parse_args()

if not os.path.isfile(args.file):
    raise ValueError("Json file does not exists.")

json_file = None
with open(args.file, "r") as f:
    json_file = json.load(f)

if args.custom:
    for elem in json_file:
        im_id = int(os.path.splitext(os.path.basename(elem['url']))[0])
        if im_id != elem['image_id']:
            print(im_id,elem['image_id'])
        if elem['image_id']==2344257:
            im = Image.open(os.path.join('/home/althausc/nfs/data/vg/VG_100K/VG_100K','%d.jpg'%elem['image_id']))
            width, height = im.size
            print(elem, im.size)
    exit(1)

if args.firstn is not None:
    print(str(json_file)[:args.firstn])
    exit(1)

if args.mode == 'sizelist':
    print("Num elem in file: ",len(json_file))
    print(json_file[len(json_file)-10:])
    exit(1)

if args.mode == 'graph-prediction':
    for elem in list(json_file['0'].items()):
        print(elem[0],np.array(elem[1]).shape)
    exit(1)

if args.mode == 'coco-annotations-firstn':
    print(str(json_file['images'])[:10000])
    print(str(json_file['annotations'])[:10000])
    exit(1)

if args.mode == 'coco-metadata':
    for ann in json_file['images']:
        if ann['id'] == int(args.search):
            print(ann)
    exit(1)

if args.mode == 'coco-annotations':
    for ann in json_file['annotations']:
        if ann['image_id'] == int(args.search):
            print(ann)
    exit(1)

if args.mode == 'coco-ann-images-number':
    print("Length of annotations: ", len(json_file['annotations']))
    print("Length of image field: ", len(json_file['images']))
    exit(1)

if args.mode == 'anns-per-imageid-stats':
    json_file.sort(key=lambda x: x['image_id'])
    from itertools import groupby
    sorted_by_imagids = groupby(json_file ,key=lambda x: x['image_id'])
    lens = [len(list(v)) for k,v in sorted_by_imagids]
    print("Average number of annotations/imagid: ", np.mean(lens))
    print("Min number of annotations/imagid: ", min(lens))
    print("Max number of annotations/imagid: ", max(lens))
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

    elif args.mode =='elem0dictnames':
        print("basekeys: ", list(json_file.keys()))
        
        first_entry = list(json_file.values())[0]
        print(type(first_entry))
        for key,val in list(first_entry.items()):
            valarray = np.array(val)
            print(key, valarray.shape)
    else:
        print("Number of entries: ",len(json_file))
        k = 9
        print("Printing first %d entries:"%k)
        for i,entry in enumerate(json_file.items()):
            print(str(entry)[:70000])
            if i==k-1:
                break





