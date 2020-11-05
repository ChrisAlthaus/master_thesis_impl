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

#Visualizes human pose keypoints given by input file.
#Treshold for score possible.
#Image folder corresponding to input file annotations needed.

def main():
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

    grouped_by_imageid = [{k: list(g)} for k, g in itertools.groupby(sorted(data, key=lambda x:x['image_id']), lambda x: x['image_id'])]
    print("Visualize the predictions onto the original image(s) ...")
    visualize(grouped_by_imageid, args.imagespath, args.vistresh, args.outputdir)
    print("Visualize done.")
    print("Wrote images to path: ",args.outputdir)

def visualize(grouped_by_imageid, imagedir, outputdir, vistresh=0.0, transformid=False, suffix='overlay'):
    #Grouped imageid input: [{imageid1 : [{imageid1,...},...,{imageid1,...}], ... , {imageidn :[{imageidn,...},...,{imageidn,...}]}]
    
    MetadataCatalog.get("my_dataset_val").set(keypoint_names=COCO_PERSON_KEYPOINT_NAMES,
                                              keypoint_flip_map=COCO_PERSON_KEYPOINT_FLIP_MAP,
                                              keypoint_connection_rules=KEYPOINT_CONNECTION_RULES)

    for pred_group in grouped_by_imageid:
        imgid = str(list(pred_group.keys())[0])
        preds = list(pred_group.values())[0]

        if transformid:
            imgname = "%s_%s.jpg"%( imgid[:len(imgid)-6].zfill(12), imgid[len(imgid)-6:])
            imgname_out = "{}_{}_{}.jpg".format(imgid[:len(imgid)-6].zfill(12), imgid[len(imgid)-6:], suffix)
            img_path = os.path.join(imagedir, imgname)
        else:
            imgname = "%s.jpg"%(imgid)
            imgname_out = "{}_{}.jpg".format(imgid, suffix)
            img_path = os.path.join(imagedir, imgname)

        img = cv2.imread(img_path, 0)
        height, width = img.shape[:2]

        instances = Instances((height, width))
        boxes = []
        scores = []
        classes = []
        masks = []
        keypoints = []

        #"image_id": 785050351, "category_id": 1, "score": 1.0, "keypoints"
        for pred in preds:
            classes.append(pred["category_id"])
            if 'score' in pred: #gt annotations don't have score entry
                scores.append(pred['score'])
            else:
                scores.append(1.0)
            kpts = list(zip(pred['keypoints'][::3], pred['keypoints'][1::3], pred['keypoints'][2::3]))
            keypoints.append(kpts)
    
        instances.scores = torch.Tensor(scores)
        instances.pred_classes = torch.Tensor(classes)
        instances.pred_keypoints = torch.Tensor(keypoints)

        v = Visualizer(cv2.imread(img_path)[:, :, ::-1],MetadataCatalog.get("my_dataset_val"), scale=1.2)
        out = v.draw_instance_predictions(instances, vistresh)

        cv2.imwrite(os.path.join(outputdir, imgname_out),out.get_image()[:, :, ::-1])

        if out == None:
            print("img is none")

def getvisualized(image, gtinstances):
    #Shape of image= (H,W,C)
    MetadataCatalog.get("my_dataset_val").set(keypoint_names=COCO_PERSON_KEYPOINT_NAMES,
                                              keypoint_flip_map=COCO_PERSON_KEYPOINT_FLIP_MAP,
                                              keypoint_connection_rules=KEYPOINT_CONNECTION_RULES)
    #Rename fields to match visualizer function
    instances = Instances(gtinstances._image_size)
    #instances.scores = torch.Tensor([])
    instances.pred_boxes = gtinstances.gt_boxes
    instances.pred_classes = gtinstances.gt_classes
    instances.pred_keypoints = gtinstances.gt_keypoints
    
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get("my_dataset_val"), scale=1.2)
    out = v.draw_instance_predictions(instances)
    outimg = out.get_image()[:, :, ::-1]
    return outimg

if __name__=="__main__":
    main()