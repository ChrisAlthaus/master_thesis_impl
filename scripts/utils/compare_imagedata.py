import argparse
import os

import datetime
import json
import random
import time

import itertools
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('-file1', help='Path to a json file.')
parser.add_argument('-file2', help='Path to a json file.')
parser.add_argument('-mode')
parser.add_argument('-search')

args = parser.parse_args()

json_file1 = None
with open(args.file1, "r") as f:
    json_file1 = json.load(f)

json_file2 = None
with open(args.file2, "r") as f:
    json_file2 = json.load(f)

"""print(json_file1[4996:5000])
print(json_file2[4996:5000])
exit(1)"""
#json_file2.sort(key= lambda x: x['image_id'])
"""start_time = time.time()
imgids1 = [x['image_id'] for x in json_file1]
json_file2.sort(key=lambda x: imgids1.index(x['image_id']))
print("--- %s seconds ---" % (time.time() - start_time))"""


if args.mode == 'compare-imageid':
    elems1 = []
    elems2 = []
    for item in json_file1:
        if item['image_id'] == int(args.search):
            elems1.append(item)
    for item in json_file2:
        if item['image_id'] == int(args.search):
            elems2.append(item)
    print("Items for image_id {} in file1:".format(args.search))
    print(elems1)
    print("Items for image_id {} in file2:".format(args.search))
    print(elems2)
    print("Number of annotations: ", len(elems1), len(elems2))
    exit(1)

if args.mode == 'compare-allimageids':
    from itertools import groupby
    #json_file1.sort(key=lambda x: x['image_id'])
    sorted_by_imagids1 = groupby(json_file1 ,key=lambda x: x['image_id'])
    #json_file2.sort(key=lambda x: x['image_id'])
    sorted_by_imagids2 = groupby(json_file2 ,key=lambda x: x['image_id'])

    g1 = []
    g2 = []
    for id1, group1 in sorted_by_imagids1:
        g1.append((id1, list(group1)))
    for id2, group2 in sorted_by_imagids2:
        g2.append((id2, list(group2)))
    print(g1)

    for id1, group1 in g1:
        for id2, group2 in g2:
            if id1 == id2:
                print("Number of annotations: ", len(group1), len(group2))
    exit(1)

#print(json_file2[:10])
for i in range(len(json_file1)):
    if json_file1[i]['image_id'] != json_file2[i]['image_id']:
        print(json_file1[i]['image_id'], json_file2[i]['image_id'])
        print(json_file1[i], json_file2[i], i)
        exit(1)
print("test")
exit(1)
ids2 = []
for i in range(len(json_file2)):
    ids2.append(json_file2[i]['image_id'])
ids1 = []
for i in range(len(json_file1)):
    ids1.append(json_file1[i]['image_id'])

print(len(ids1),len(set(ids1)))
print(len(ids2),len(set(ids2)))

print(np.setdiff1d(ids1, ids2)) #Elements of ids1 which are not in ids2
print(np.setdiff1d(ids2, ids1)) #Elements of ids2 which are not in ids1

print(json_file2[4996],json_file2[2])