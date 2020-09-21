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