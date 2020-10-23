import argparse
import os

from pycocotools.coco import COCO
import cv2
from scipy.spatial import distance
import datetime
import json
import random
import time
import copy

import itertools
import numpy as np
from shapely.geometry import Polygon
import sys

visualize_dir = '/home/althausc/master_thesis_impl/scripts/utils'
sys.path.insert(0,visualize_dir) 

import visualizekpts as vkpts

parser = argparse.ArgumentParser()
parser.add_argument('-prediction_path','-predictions', 
                    help='Path to the prediction json file.')
parser.add_argument('-gt_annotations','-annotations', 
                    help='Path to the corresponding json gt annotations.')
parser.add_argument('-visfirstn',action="store_true", 
                        help='Wheather to visualize single & merged predictions.')   
parser.add_argument('-drawbmapping',action="store_true", 
                        help='Wheather to visualize bbox mapping.')   
args = parser.parse_args()
if not os.path.isfile(args.prediction_path):
    raise ValueError("Prediction file does not exists.")
if not os.path.isfile(args.gt_annotations):
    raise ValueError("Annotation file does not exists.")

#For visualization
_IMGDIR = '/home/althausc/nfs/data/coco_17_medium/val2017_styletransfer'
_VIS_NUM = 20
#For mapping gt anns-> predictions
_IOU_TRESH = 0.5

def main():   
    #Preprocessing: search for every prediction the most probable corresponding gt bounding box
    #               in the annotation file
    #   -> Keypoints of the predictions should be retained, whereas bboxes should be replaced by ground truth

    with open(args.prediction_path, "r") as f:
        preds = json.load(f)
    with open(args.gt_annotations, "r") as f:
        annotations = json.load(f)['annotations']
    

    output_dir = os.path.join('/home/althausc/master_thesis_impl/PoseFix_RELEASE/inputs', datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        raise ValueError("Output directory %s for checkpoints already exists. Please wait a few minutes."%output_dir)


    preds = sorted(preds, key=lambda k: k['image_id']) 
    annotations = sorted(annotations, key=lambda k: k['image_id'])

    #Sort predictions into dicts with image_id as key
    #[{image_id1: pred1, pred4}, ...]
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
        #Annotation bbox format not equal to prediction format, therefore transform
        x,y,w,h = ann['bbox'] #x, y= top left coordinates ; w,h = width & height
        ann['bbox'] = [x,y,x+w,y+h]
        if image_id in preds_group:
            if image_id in annotations_group:
                annotations_group[image_id].append(ann)
            else:
                annotations_group.update({image_id:[ann]})
    
    numimgids_intersect = len(set( list(annotations_group.keys()) ).intersection( list(preds_group.keys()) ))

    #For each gt annontation get most probable prediction annotation based on overlab of bounding boxes
    #If overlab smaller than treshold, take the gt annotation
    #If too few predictions, take the remaining entries from the gt annotations
    result = []
    groups_sample = []
    bmapping_sample = [] #[(boxann,box_pred), ...] for debugging
    stats = {'numann':0 ,'numpred':0, 'nummap':0, 'numidentity':0}
    c = 0

    for image_id, ann_group in annotations_group.items():
        if image_id in preds_group:
            pred_group = copy.deepcopy(preds_group[image_id])
            merged = []
            bmapping = {'image_id': str(image_id), 'bmaps': [], 'bpred': [pred['bbox'] for pred in pred_group], 'bann': [ann['bbox'] for ann in ann_group]}

            stats['numann'] =  stats['numann'] + len(ann_group)
            stats['numpred'] =  stats['numpred'] + len(pred_group)

            for i,ann_gt in enumerate(ann_group):
                b_ann = ann_gt['bbox']
                if len(pred_group) == 0:
                    merged.append(ann_gt)
                    stats['numidentity'] = stats['numidentity'] + 1
                    continue
                
                b_pred = [pred['bbox'] for pred in pred_group]
                sim = [calculate_iou(b_ann, b) for b in b_pred]
                max_index = np.argmax(sim)

                if sim[max_index] >= _IOU_TRESH:
                    pred = pred_group[max_index]
                    entry = {'image_id': pred['image_id'], 'image_size':  pred['image_size'], 'category_id': pred['category_id'], 
                            'bbox': b_ann, 'keypoints': pred['keypoints'], 'score': pred['score'] }
                    merged.append(entry)
                    bmapping['bmaps'].append((b_ann,pred['bbox']))
                    stats['nummap'] = stats['nummap'] + 1

                else:
                    merged.append(ann_gt)
                    bmapping['bmaps'].append([b_ann])
                    stats['numidentity'] = stats['numidentity'] + 1
                del pred_group[max_index]

            result.extend(merged)
            #Save visualize annotations
            if len(groups_sample) < _VIS_NUM:
                groups_sample.append({'pred': [{image_id: preds_group[image_id]}], 'ann': [{image_id: ann_group}], 'merged':[{image_id: merged}] })
            #Save mapping annotations
            if len(bmapping_sample) < _VIS_NUM:
                bmapping_sample.append(bmapping)

            c = c + 1
            #if c==100:
            #    break    
    
    statistics_str = 'Number of unique annotated imageids: %d'%len(annotations_group) + os.linesep
    statistics_str = statistics_str + 'Number of unique prediction imageids: %d'%len(preds_group) + os.linesep
    statistics_str = statistics_str + "Number of intersection of imageids: %d"%numimgids_intersect + os.linesep

    statistics_str = statistics_str + "Length gt annotations: %d"%len(annotations) + os.linesep
    statistics_str = statistics_str + "Length of predictions: %d"%len(preds) + os.linesep
    statistics_str = statistics_str + "Length of filtered & combined predictions: %d"%len(result) + os.linesep
    statistics_str = statistics_str + "Length of filtered & combined predictions (unique imageid): %d"%c + os.linesep
    
    perc_mapped = stats['nummap']/stats['numann']
    perc_identity = stats['numidentity']/stats['numann']

    statistics_str = statistics_str + "Fraction of gt annotations mapped to predictions: %f"%perc_mapped + os.linesep
    statistics_str = statistics_str + "Fraction of gt annotations mapped to identity/added: %f"%perc_identity + os.linesep
    statistics_str = statistics_str + "Number of considered annotations:" + os.linesep
    statistics_str = statistics_str + "\t prediction: %d"%stats['numpred'] + os.linesep
    statistics_str = statistics_str + "\t groundtruth: %d"%stats['numann'] + os.linesep

    print(statistics_str)
    #Save statistics to config 
    with open(os.path.join(output_dir, 'config.txt'), 'w') as f:   
        f.write("Annotation file: %s"%args.gt_annotations + os.linesep)
        f.write("Prediction file: %s"%args.prediction_path + os.linesep)
        f.write(os.linesep)

        f.write(statistics_str)


    print("Output folder: ", output_dir)
    with open(os.path.join(output_dir,'predictions_bbox_gt.json'), 'w') as f:
        json.dump(result, f, separators=(', ', ': '))  


    if args.visfirstn or args.drawbmapping:
        visdir = os.path.join(output_dir, '.visimages')
        if not os.path.exists(visdir):
            os.makedirs(visdir)   

    if args.visfirstn:
        print("Visualize for validation ...")
        #Visualize for validation purposes
        visdir = os.path.join(output_dir, '.visimages')
        if not os.path.exists(visdir):
            os.makedirs(visdir)
        for sample in groups_sample:
            vkpts.visualize(sample['pred'], _IMGDIR, visdir, vistresh=0.0, transformid=True, suffix='pred')
            vkpts.visualize(sample['ann'], _IMGDIR, visdir, vistresh=0.0, transformid=True, suffix='gtann')
            vkpts.visualize(sample['merged'], _IMGDIR, visdir, vistresh=0.0, transformid=True, suffix='merged')
        print("Visualize for validation done.")
        print("Wrote images to: ",visdir)   

    if args.drawbmapping:
        print("Drawing bounding box mapping ...")
        for bm in bmapping_sample:
            draw_bboxmapping(bm['image_id'], _IMGDIR, visdir, bm['bmaps'], bm['bpred'], bm['bann'], transformid=True)
        print("Drawing bounding box mapping done.")
        print("Wrote images to: ",visdir)
    
def getbboxes(list_of_dicts):
    bbox_list = []
    for entry in list_of_dicts:
        bbox = entry['bbox']
        bbox_list.append(bbox)
    return bbox_list

def calculate_iou(box_1, box_2):
    box_1 = [int(x) for x in box_1]
    box_2 = [int(x) for x in box_2]
    box_1 = list(zip(box_1[::2], box_1[1::2]))
    box_2 = list(zip(box_2[::2], box_2[1::2]))

    def getother2points(points):
        p1,p2 = points
        p3 = p1[0],p2[1]
        p4 = p2[0],p1[1]
        return [p1,p3,p2,p4]

    box_1 = getother2points(box_1)
    box_2 = getother2points(box_2)
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)

    if poly_1.intersects(poly_2):
        iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
        return iou
    return 0.0

def draw_bboxmapping(imgid, imagedir, outputdir, mapping, bpred, bann, transformid=False):
    if transformid:
        imgname = "%s_%s.jpg"%( imgid[:len(imgid)-6].zfill(12), imgid[len(imgid)-6:])
        imgname_out = "{}_{}_bmapping.jpg".format(imgid[:len(imgid)-6].zfill(12), imgid[len(imgid)-6:])
        img_path = os.path.join(imagedir, imgname)
    else:
        imgname = "%s.jpg"%(imgid)
        imgname_out = "{}_bmapping.jpg"%(imgid)
        img_path = os.path.join(imagedir, imgname)
    img = cv2.imread(img_path, 1)

    def getpoints(bbox):
        bbox = list(map(int, bbox))
        p1 = (bbox[0], bbox[1])
        p2 = (bbox[2], bbox[3])
        return p1,p2

    for b1 in bpred:
        p1,p2 = getpoints(b1)
        cv2.rectangle(img, p1, p2, (0,200,200), (4)) #yellow

    for b2 in bann:
        p1,p2 = getpoints(b2)
        cv2.rectangle(img, p1, p2, (255,0,0), (4)) #blue

    for m in mapping:
        if len(m)==1: 
            p1,p2 = getpoints(m[0])
            cv2.rectangle(img, p1, p2, (0,0,255), (1)) #red
        else:
            p11,p12 = getpoints(m[0])
            p21,p22 = getpoints(m[1])
            cv2.rectangle(img, p11, p12, (0,0,255), (1)) #red
            cv2.rectangle(img, p21, p22, (0,0,255), (1)) #red
            cv2.arrowedLine(img, p11, p21, (255,255,255), (2), 8,0,0.25) #white
    
    cv2.imwrite(os.path.join(outputdir, imgname_out), img)

if __name__=="__main__":
    main()