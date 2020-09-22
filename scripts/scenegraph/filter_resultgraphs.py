import argparse
import os

import datetime
import json
import random
import time

import itertools
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('-file', help='Path to the prediction graph file.')
parser.add_argument('-imginfo', help='Path to the image info file.')
parser.add_argument('-outputdir')

args = parser.parse_args()

if not os.path.isfile(args.file) or not os.path.isfile(args.imginfo):
    raise ValueError("Json file does not exist.")
if not os.path.isdir(args.outputdir):
    raise ValueError("Output directory does not exist.")

output_dir = os.path.join(args.outputdir, datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
else:
    raise ValueError("Output directory %s already exists."%output_dir)

data = None
with open(args.file, "r") as f:
    data = json.load(f)

imginfo = None
with open(args.imginfo, "r") as f:
    imginfo = json.load(f)

#Graph file format:
#   every dict item represents one image prediction
#       ->fields are:   bbox (80, 4)    (sizes may differ)
#                       bbox_labels (80,)
#                       bbox_scores (80,)
#                       rel_pairs (6320, 2)
#                       rel_labels (6320,)
#                       rel_scores (6320,)
#                       rel_all_scores (6320, 51)

idx_to_files = imginfo["idx_to_files"]
ind_to_classes = imginfo["ind_to_classes"]
ind_to_predicates = imginfo["ind_to_predicates"]
_SCORE_THRESH_BOXES = 0.01
_SCORE_THRESH_RELS = 0.01

boxes_topk = 10
rels_topk = 20
graphs = dict()

for idx, preds in list(data.items()):
    boxes = preds['bbox'][:boxes_topk]
    b_labels = preds['bbox_labels'][:boxes_topk]
    b_scores = preds['bbox_scores'][:boxes_topk]
    #TODO: save scores in other file for search ranking
    
    k=boxes_topk
    for i,s in enumerate(b_scores):
        if s<_SCORE_THRESH_BOXES:
            k=i
            break

    boxes = boxes[:k]
    b_labels = b_labels[:k]
    b_scores = b_scores[:k]
    b_indices = list(range(k))

    b_classes = []
    for l in b_labels:
        b_classes.append(ind_to_classes[l])


    rels = preds['rel_pairs']
    rel_scores = preds['rel_scores']
    rels_valid = []
    rels_scores = []
    for i,rel in enumerate(rels):
        if rel[0] in b_indices and rel[1] in b_indices:
            if rel_scores[i] >= _SCORE_THRESH_RELS:
                rels_valid.append(rel)
                rels_scores.append(rel_scores[i])



    graph = {'edges': rels_valid, 'features': {ind:c_label for ind,c_label in list(zip(b_indices,b_classes))}}
    graph.update({'box_scores': {ind:b_score for ind,b_score in list(zip(b_indices,b_scores))}})
    graph.update({'rel_scores': rels_scores})

    graphs[idx_to_files[int(idx)]] = graph


print("Number of graphs in output file: ",len(graphs))
outfile_name = "graphs-topk.json"
with open(os.path.join(output_dir, outfile_name), 'w') as f:
    print("Writing to file: ",os.path.join(output_dir, outfile_name))
    json.dump(graphs, f)   
