import argparse
import os

from pycocotools.coco import COCO
import datetime
import json
import random
import time

import itertools
import numpy as np

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#tips = sns.load_dataset("tips")
#print(tips)
#exit(1)
parser = argparse.ArgumentParser()
parser.add_argument('-file', help='Path to a json file.')
parser.add_argument('-gtanno', help='Path to the groundtruth annotations.')

args = parser.parse_args()

if not os.path.isfile(args.file):
    raise ValueError("Json file does not exists.")

json_file = None
with open(args.file, "r") as f:
    json_file = json.load(f)
ann_file = None
with open(args.gtanno, "r") as f:
    ann_file = json.load(f)

_SCORE_THRESH = 0.9
_NUMKPTs = 17

#Create box plot to visualize the distribution of visibilities/probs for each keypoint category
visibilities = [[] for x in range(_NUMKPTs)] 

k = 1
for entry in json_file:
    if entry['score'] >= _SCORE_THRESH:
        for i in range(_NUMKPTs):
            visibilities[i].append(entry['keypoints'][(i+1)*3-1])

for i in range(_NUMKPTs):
    print("Mean %d: "%i,np.mean(visibilities[i]))

data = dict()
for i,visbs in enumerate(visibilities):
    data.update({str(i):visbs})
df_kpts_visbs = pd.DataFrame(data)
print(df_kpts_visbs)
print(df_kpts_visbs[df_kpts_visbs.columns[0]].max())
ax = df_kpts_visbs.boxplot()
ax.set_yscale("symlog",linthreshy=1)
ax.set_title("Probability distribution",fontsize=16)
ax.set_xlabel("Keypoint category")
ax.set_ylabel("Confidences")

ax.figure.savefig("kpts_visibility_stats.png")

#Create bar plot with number of predictions vs gt annoations
annotation = ann_file['annotations']
num_anns = len(annotation)
num_preds = len(json_file)

df_numelems = pd.DataFrame({'gt annotations': [num_anns], 'predictions':[num_preds]})
ax = sns.barplot(data=df_numelems)
ax.figure.savefig("predann_numelements.png")