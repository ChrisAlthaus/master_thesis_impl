
import argparse
import os

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

from validlabels import ind_to_classes, ind_to_predicates, VALID_BBOXLABELS, VALID_RELLABELS


parser = argparse.ArgumentParser()
parser.add_argument('-graphfile', help='Path to the filtered graph file.')
args = parser.parse_args()

outdir = os.path.dirname(args.graphfile)



data = None
with open(args.graphfile, "r") as f:
    data = json.load(f)

graphs = list(data.items()) #[(filename, graph) ... , (filename, graph)]
                            #graph = {'edges': ... , 'features': ... , 'box_scores': ... , 'rel_scores': ...}


bbox_stats = [0] * len(VALID_BBOXLABELS)
rel_stats = [0] * len(VALID_RELLABELS)
bbox_scores = []
rel_scores = []

for filename, g in graphs:
    for item in list(g['features'].items()):
        if item[1] in VALID_BBOXLABELS:
            bbox_stats[VALID_BBOXLABELS.index(item[1])] = bbox_stats[VALID_BBOXLABELS.index(item[1])] + 1
            bbox_scores.append([ item[1],g['box_scores'][item[0]] ])
        elif item[1] in VALID_RELLABELS:
            rel_stats[VALID_RELLABELS.index(item[1])] = rel_stats[VALID_RELLABELS.index(item[1])] + 1
            rel_scores.append([item[1], g['rel_scores'][item[0]] ])
        else:
            print("Node class not valid.")

#BOX AND REL CLASSES PLOT
a4_dims = (30, 15)
fig, ax = plt.subplots(figsize=a4_dims)
df_bbox = pd.DataFrame({'Box Labels': VALID_BBOXLABELS, 'Quantity':bbox_stats})
df_bbox = df_bbox.sort_values('Quantity', ascending=False).reset_index()
ax = sns.barplot(x='Box Labels', y='Quantity', data=df_bbox)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
ax.figure.savefig(os.path.join(outdir, 'bbox_statistics.jpg'))
plt.clf()

a4_dims = (30, 15)
fig, ax = plt.subplots(figsize=a4_dims)
df_rels = pd.DataFrame({'Relation Labels': VALID_RELLABELS, 'Quantity':rel_stats})
df_rels = df_rels.sort_values('Quantity', ascending=False).reset_index()
ax = sns.barplot(x='Relation Labels', y='Quantity', data=df_rels)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
ax.figure.savefig(os.path.join(outdir, 'rels_statistics.jpg'))
plt.clf()


#PLOT BOX AND RELS PREDICTION SCORE DISTRIBUTION PER LABEL
index = df_bbox['Box Labels'].to_list()
sorterIndex = dict(zip(index, range(len(index))))
bbox_scores = sorted(bbox_scores, key=lambda x: sorterIndex[x[0]])

df_scores = pd.DataFrame(bbox_scores, columns=('Box Labels', 'Scores'))  

print(df_scores) 
ax = sns.boxplot(x='Box Labels', y='Scores', 
                 data=df_scores)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
ax.figure.savefig(os.path.join(outdir, 'bbox_score_stats.jpg'))
plt.clf()

index = df_rels['Relation Labels'].to_list()
sorterIndex = dict(zip(index, range(len(index))))
rel_scores = sorted(rel_scores, key=lambda x: sorterIndex[x[0]])

df_relscores = pd.DataFrame(rel_scores, columns=('Relation Labels', 'Scores'))   
ax = sns.boxplot(x='Relation Labels', y='Scores', 
                 data=df_relscores)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
ax.figure.savefig(os.path.join(outdir, 'rels_score_stats.jpg'))
plt.clf()