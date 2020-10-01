import argparse
import os

from pycocotools.coco import COCO
import cv2
from scipy.spatial import distance
import datetime
import json
import random
import time

import itertools
import numpy as np



parser = argparse.ArgumentParser()
parser.add_argument('-prediction_path','-predictions', 
                    help='Path to the prediction json file.')
parser.add_argument('-gt_annotations','-annotations', 
                    help='Path to the corresponding json gt annotations.')

args = parser.parse_args()

if not os.path.isfile(args.prediction_path):
    raise ValueError("Prediction file does not exists.")
if not os.path.isfile(args.gt_annotations):
    raise ValueError("Annotation file does not exists.")


#Preprocessing: search for every prediction the most probable corresponding gt bounding box
#               in the annotation file
#   -> Keypoints of the predictions should be retained, whereas bboxes should be replaced by ground truth

with open(args.prediction_path, "r") as f:
    preds = json.load(f)
with open(args.gt_annotations, "r") as f:
    annotations = json.load(f)['annotations']

output_dir = os.path.join('/home/althausc/master_thesis_impl/PoseFix_RELEASE/inputs', datetime.datetime.now().strftime('%m_%H-%M-%S'))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
else:
    raise ValueError("Output directory %s for checkpoints already exists. Please wait a few minutes."%output_dir)


preds = sorted(preds, key=lambda k: k['image_id']) 
annotations = sorted(annotations, key=lambda k: k['image_id'])

#preds = list(itertools.groupby(preds, lambda x: x['image_id']))
#annotations = list(itertools.groupby(annotations, lambda x: x['image_id']))

preds_group = dict()
for pred in preds:
    image_id = pred['image_id']
    if image_id in preds_group:
        preds_group[image_id].append(pred)
    else:
        preds_group.update({image_id:[pred]})

annotations_group = dict()
for ann in annotations:
    image_id = ann['image_id']
    if (np.sum(ann['keypoints'][2::3]) == 0) or (ann['num_keypoints'] == 0):
        continue
    if image_id in preds_group:
        if image_id in annotations_group:
            annotations_group[image_id].append(ann)
        else:
            annotations_group.update({image_id:[ann]})

def getbboxes(list_of_dicts):
    bbox_list = []
    for entry in list_of_dicts:
        bbox = entry['bbox']
        bbox_list.append(bbox)
    return bbox_list

print("Replacing bbox of predictions with groundtruth annotation...")

preds_with_gtbbox = []
for img_id, pred_group in preds_group.items():
    if img_id in annotations_group:
        ann_group = annotations_group[img_id]
        b1 = getbboxes(pred_group)
        b2 = getbboxes(ann_group)

        dist = distance.cdist(b2, b1, 'euclidean')
        for i,row in enumerate(dist):
            min_index = np.argmin(row)

            #print("MIN INDEX",pred_group[min_index], "TYPE ",type(pred_group[min_index]))
            #print("UPDATE",ann_group[i]['bbox'])
            pred_group[min_index]['bbox'] =  ann_group[i]['bbox']
            
            preds_with_gtbbox.append( pred_group[min_index])
            
            #prediction should not be considered again
            for row in dist:
                row[min_index] = 10e04

print("Replacing bbox of predictions with groundtruth annotation done.")
print("Length gt annotations: ",len(annotations))
print("Length of filtered & combined predictions: ",len(preds_with_gtbbox))
print("Length of previous predictions: ",len(preds))


with open(os.path.join(output_dir,'predictions_bbox_gt.json'), 'w') as f:
    json.dump(preds_with_gtbbox, f, separators=(', ', ': '))    
