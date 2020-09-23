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
parser.add_argument('-boxestresh', type=float, default=0.01)
parser.add_argument('-relstresh', type=float, default=0.01)
parser.add_argument('-build_labelvectors', action='store_true')

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
_SCORE_THRESH_BOXES = args.boxestresh
_SCORE_THRESH_RELS = args.relstresh

boxes_topk = 20
rels_topk = 20
graphs = dict()
ann = ''

for idx, preds in list(data.items()):
    boxes = preds['bbox'][:boxes_topk]
    b_labels = preds['bbox_labels'][:boxes_topk]    #assumption: model just supports valid labels
    b_scores = preds['bbox_scores'][:boxes_topk]    
    #TODO: save scores in other file for search ranking
    #TODO: filter labels 
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
    
    for i,l in enumerate(b_labels):
        b_classes.append(ind_to_classes[l])
        ann = ann + str(i) + '_' + ind_to_classes[l] +'; '+ str(b_scores[i]) + '\n'


    rels = preds['rel_pairs']
    rel_scores = preds['rel_scores']
    rel_labels = preds['rel_labels']
    rels_valid = []
    rels_scores = []
    for i,rel in enumerate(rels):
        if rel[0] in b_indices and rel[1] in b_indices:
            if rel_scores[i] >= _SCORE_THRESH_RELS:
                rels_valid.append(rel)
                rels_scores.append(rel_scores[i])

                b1_label = ind_to_classes[b_labels[rel[0]]]
                b2_label = ind_to_classes[b_labels[rel[1]]]
                relstr = str(rel[0]) + '_' + b1_label + ' => ' + ind_to_predicates[rel_labels[i]] + ' => ' + str(rel[1]) + '_' + b2_label
                relstr = relstr + '; '+ str(rel_scores[i])
                ann = ann + relstr + '\n'

    print("Annotations filtered (for validation purposes): \n", ann)

    graph = {'edges': rels_valid, 'features': {ind:c_label for ind,c_label in list(zip(b_indices,b_classes))}}
    graph.update({'box_scores': {ind:b_score for ind,b_score in list(zip(b_indices,b_scores))}})
    graph.update({'rel_scores': rels_scores})

    graphs[idx_to_files[int(idx)]] = graph

if len(data) == 1:
    annname = "labels.txt"
    with open(os.path.join(output_dir, annname), "w") as text_file:
        text_file.write(ann)


print("Number of graphs in output file: ",len(graphs))
outfile_name = "graphs-topk.json"
with open(os.path.join(output_dir, outfile_name), 'w') as f:
    print("Writing to file: ",os.path.join(output_dir, outfile_name))
    json.dump(graphs, f)   


from validlabels import VALID_BBOXLABELS, VALID_RELLABELS

if args.build_labelvectors:
    labelvectors = []
    for idx, preds in list(data.items()):
        #sanity check for valid labels
        b_labels = []
        for l in preds['bbox_labels']:
            if ind_to_classes[l] in VALID_BBOXLABELS:
                b_labels.append(l)
            if len(b_labels) >= boxes_topk:
                break
        
        rel_labels = []
        for r in preds['rel_labels']:
            if ind_to_predicates[r] in VALID_RELLABELS:
                rel_labels.append(r)
            if len(rel_labels) >= rels_topk:
                break

        box_labelvector = np.zeros(len(VALID_BBOXLABELS))
        for l in b_labels:
            b_class = ind_to_classes[l]
            box_labelvector[VALID_BBOXLABELS.index(b_class)] = box_labelvector[VALID_BBOXLABELS.index(b_class)] + 1

        rels_labelvector = np.zeros(len(VALID_RELLABELS))
        for r in rel_labels:
            r_class = ind_to_predicates[r]
            rels_labelvector[VALID_RELLABELS.index(r_class)] = rels_labelvector[VALID_RELLABELS.index(r_class)] + 1
 
        labelvectors.append({'imagefile': idx_to_files[int(idx)], 'boxvector': box_labelvector.tolist(), 'relvector': rels_labelvector.tolist()})

    print("Number of labelvectors in output file: ",len(data))
    outfile_name = "labelvectors-topk.json"
    with open(os.path.join(output_dir, outfile_name), 'w') as f:
        print("Writing to file: ",os.path.join(output_dir, outfile_name))
        json.dump(labelvectors, f)  

    



