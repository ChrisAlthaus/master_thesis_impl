import time
import argparse
import json
import logging
from detectron2.structures import Instances
from detectron2.structures import Boxes
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.builtin_meta import KEYPOINT_CONNECTION_RULES, COCO_PERSON_KEYPOINT_NAMES, COCO_PERSON_KEYPOINT_FLIP_MAP
import os
import cv2
import torch
import itertools


parser = argparse.ArgumentParser()
parser.add_argument('-file',required=True,
                    help='File with a list of dict items with fields imageid and keypoints.')
parser.add_argument('-imagespath',required=True)    
parser.add_argument('-outputdir',required=True) 
parser.add_argument('-transformid',action="store_true", 
                    help='Wheather to split imageid to get image filepath (used for style transfered images.')    
parser.add_argument('-vistresh',type=float, default=0.0)                
args = parser.parse_args()
    
data = None
with open(args.file, "r") as f:
    data = json.load(f)   

grouped_by_imageid = [list(g) for k, g in itertools.groupby(sorted(data, key=lambda x:x['image_id']), lambda x: x['image_id'])]
MetadataCatalog.get("my_dataset_val").set(keypoint_names=COCO_PERSON_KEYPOINT_NAMES,
                                          keypoint_flip_map=COCO_PERSON_KEYPOINT_FLIP_MAP,
                                          keypoint_connection_rules=KEYPOINT_CONNECTION_RULES)

print("Visualize the predictions onto the original image(s) ...")
for preds_imgs in grouped_by_imageid:
    imgid = str(preds_imgs[0]['image_id'])
    
    #if args.transformid: #style-transfered image #deprecated ?!
    #    imgname = "%s_%s.jpg"%( imgid[:len(imgid)-6].zfill(12), imgid[len(imgid)-6:])
    #    imgname_out = "%s_%s_overlay.jpg"%( imgid[:len(imgid)-6].zfill(12), imgid[len(imgid)-6:])
    #    img_path = os.path.join(args.imagespath, imgname)
    #else:
    imgname = "%s.jpg"%(imgid)
    imgname_out = "%s_overlay.jpg"%(imgid)
    img_path = os.path.join(args.imagespath, imgname)

    img = cv2.imread(img_path, 0)
    height, width = img.shape[:2]

    instances = Instances((height, width))
    boxes = []
    scores = []
    classes = []
    masks = []
    keypoints = []
    #"image_id": 785050351, "category_id": 1, "score": 1.0, "keypoints"
    for pred in preds_imgs:
        classes.append(pred["category_id"])
        scores.append(pred['score'])
        kpts = list(zip(pred['keypoints'][::3], pred['keypoints'][1::3], pred['keypoints'][2::3]))
        keypoints.append(kpts)
    
    instances.scores = torch.Tensor(scores)
    instances.pred_classes = torch.Tensor(classes)
    instances.pred_keypoints = torch.Tensor(keypoints)

    v = Visualizer(cv2.imread(img_path)[:, :, ::-1],MetadataCatalog.get("my_dataset_val"), scale=1.2)
    out = v.draw_instance_predictions(instances, args.vistresh)
     
    cv2.imwrite(os.path.join(args.outputdir, imgname_out),out.get_image()[:, :, ::-1])

    print("Wrote image to path: ",os.path.join(args.outputdir, imgname_out))

    if out == None:
        print("img is none")
    
print("Visualize done.")