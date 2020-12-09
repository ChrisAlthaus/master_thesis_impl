import argparse
import os

import datetime
import json
import random
import time

import itertools
import numpy as np

from validlabels import ind_to_classes, ind_to_predicates, VALID_BBOXLABELS, VALID_RELLABELS

#Can be used for scene graph prediction topk filtering before saving
#Otherwise much too large (~7MB/prediction)
#Used in: Scene-Graph-Benchmark.pytorch/maskrcnn_benchmark/engine/inference.py

#Filter the graph predictions by:
#   1.bbox score treshold
#   2.valid bbox label
#   3.take topk of these filtered
#Filter the relationship predictions by:
#   1.rel score treshold
#   2.valid rel label
#   3.take topk of these filtered

def get_topkpredictions(preds, topk_boxes, topk_rels, filtertresh_boxes, filtertresh_rels):   
    #  Example pred format:
    #  bbox (80, 4)    (sizes may differ)
    #  bbox_labels (80,)
    #  bbox_scores (80,)
    #  rel_pairs (6320, 2)
    #  rel_labels (6320,)
    #  rel_scores (6320,)
    #  rel_all_scores (6320, 51)

    _SCORE_THRESH_BOXES = filtertresh_boxes
    _SCORE_THRESH_RELS = filtertresh_rels

    _BOXES_TOPK = topk_boxes
    _RELS_TOPK = topk_rels

    boxes = preds['bbox']
    box_labels = preds['bbox_labels']   #assumption: model just supports valid labels
    box_scores = preds['bbox_scores'] 

    _DEBUG = False#True

    #filter out unvalid boxes
    b_data = []
    b_labels = []
    b_scores = []
    b_indices = [] 
    skipped_indices = []
    for i, l in enumerate(box_labels):
        if ind_to_classes[l] in VALID_BBOXLABELS:
            if box_scores[i]>=_SCORE_THRESH_BOXES:
                b_indices.append(i)
                
                b_data.append(boxes[i])
                b_labels.append(l)
                b_scores.append(box_scores[i])
                if _DEBUG:
                    print("Add bbox of class {} ,score = {}".format(ind_to_classes[l], box_scores[i]))
            else:
                skipped_indices.append(i)
                if _DEBUG:
                    print("Delete bbox of class {} ,score = {}".format(ind_to_classes[l], box_scores[i]))
        else:
            skipped_indices.append(i)
            if _DEBUG:
                print("Delete bbox of class {} because not in valid labels".format(ind_to_classes[l], box_scores[i]))
        if len(b_labels) >= _BOXES_TOPK and _BOXES_TOPK != -1:
            break
        
    #print("Number of valid boxes: ",b_labels)
    
    rels = preds['rel_pairs']
    rel_labels = preds['rel_labels']
    rel_scores = preds['rel_scores']
    
    print("Pevious number of boxes: {}, Reduced number of boxes: {}".format(len(box_labels), len(b_data)))
    #filter out unvalid relations
    r_data = []
    r_scores = []
    r_labels = []
    delc = 0
    for i,rel in enumerate(rels):
        if rel[0] in b_indices and rel[1] in b_indices:
            if rel_scores[i] >= _SCORE_THRESH_RELS:
                r_data.append(rel)
                r_scores.append(rel_scores[i])
                r_labels.append(rel_labels[i])
                if _DEBUG:
                    print("Add rel of class {} ,score = {}".format(ind_to_predicates[rel_labels[i]], rel_scores[i]))
            else:
                if _DEBUG:
                    print("Delete rel of class {} ,score = {}".format(ind_to_predicates[rel_labels[i]], rel_scores[i]))
                delc = delc + 1
        else:
            if _DEBUG:
                print("Delete rel of class {} ,because of delted bboxes".format(ind_to_predicates[rel_labels[i]], rel_scores[i]))
            delc = delc + 1
        if len(r_data)>_RELS_TOPK and _RELS_TOPK != -1:
            break

    print("Pevious number of rels: {}, Reduced number of rels: {}".format(len(rels), len(r_data)))   

    #align rels indices to match the new box ordering
    b_indices_map = dict()
    for i,b_ind in enumerate(b_indices):
        b_indices_map[b_ind] = i
    #print("b_indices_map: ",b_indices_map)
    #print("r_data: ",r_data)
    for rel in r_data:
        rel[0] = b_indices_map[rel[0]]
        rel[1] = b_indices_map[rel[1]]   

    #print("r_data: ",r_data)
    """for i in skipped_indices:
        for rel in r_data:
            if rel[0] > i:
                rel[0] = rel[0] - 1
            if rel[1] > i:
                rel[1] = rel[1] - 1
    print("skipped: ",skipped_indices)
    print("r_data: ",r_data)"""

    pred_filtered = {'bbox': b_data, 'bbox_labels': b_labels, 'bbox_scores': b_scores, 
                    'rel_pairs': r_data, 'rel_labels': r_labels, 'rel_scores': r_scores}

    return pred_filtered
    

    



