from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.builtin_meta import KEYPOINT_CONNECTION_RULES, COCO_PERSON_KEYPOINT_NAMES, COCO_PERSON_KEYPOINT_FLIP_MAP
from detectron2.data import detection_utils as utils
from detectron2.data.datasets.coco import load_coco_json
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.samplers import InferenceSampler
import itertools, copy
from detectron2.data.build import build_detection_test_loader
from detectron2.structures import Instances
from detectron2.structures.boxes import Boxes

from detectron2.data.datasets import register_coco_instances
import torch
import numpy as np

import argparse
import os

from pycocotools.coco import COCO
import cv2
import datetime
import json
import random


parser = argparse.ArgumentParser()
parser.add_argument('-model_cp','-mc',required=True, 
                    help='Path to the model checkpoint file.')
parser.add_argument('-image_path','-img', 
                    help='Path to the image for which pose inference will be calculated.')
parser.add_argument('-image_folder','-imgdir', 
                    help='Path to a directory containing images only.')
parser.add_argument('-topk', type=int, default=20,
                    help='Filter the predictions and take best k poses.')
parser.add_argument('-score_tresh', type=float, default=0.5,
                    help='Filter detected poses based on a score treshold.')
parser.add_argument('-vis','-visualize', action='store_true',
                    help='Specify to visualize predictions on images & save.')
parser.add_argument('-visrandom','-validate', action='store_true',
                    help='Specify to randomy visualize k predictions.')
parser.add_argument('-vistresh', type=float, default=0.0,   
                    help='Specify a treshold for visualization.')   #not used
parser.add_argument('-transformid',action="store_true", 
                    help='Wheather to tranform image name to style-transform image id (used for style transfered images.')   
parser.add_argument('-target',
                    help='Whether to later use predictions for training other model or for querying.\
                         Output folder will then be different (train/single).')     
args = parser.parse_args()


if not os.path.isfile(args.model_cp):
    raise ValueError("Model file path not exists.")
if args.image_path is not None:
    if not os.path.isfile(args.image_path):
        raise ValueError("Image does not exists.")
if args.image_folder is not None:
    if not os.path.isdir(args.image_folder):
        raise ValueError("Image does not exists.")

if args.image_path is None and args.image_folder is None:
    raise ValueError("Please specify an image or an image directory.")

if args.target not in ['train', 'query']:
    raise ValueError("Please specify a valid prediction purpose.")
if args.score_tresh > 1 or args.score_tresh < 0:
    raise ValueError("Please specify a valid filter treshold.")

#cfg = get_cfg()

#cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))
#cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
#cfg.MODEL.DEVICE='cpu'
#model = build_model(cfg) 
#DetectionCheckpointer(model).load(args.model_cp) 
image_paths = None
if args.image_folder is not None:
    image_paths = [os.path.join(args.image_folder, x) for x in os.listdir(args.image_folder)]
elif args.image_path is not None:
    image_paths = [args.image_path]


#Register to DatasetCatalog and MetadataCatalog
#register_coco_instances("my_dataset_val", {},"/home/althausc/nfs/data/coco_17/annotations/person_keypoints_val2017.json", "/home/althausc/nfs/data/coco_17/val2017")


#model = build_model(cfg) 
#model.eval()

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")) 
#cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.DEVICE= 'cuda' #'cpu'
cfg.MIN_SIZE_TRAIN= 512
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml") 
cfg.MODEL.WEIGHTS = args.model_cp #uncomment for default checkpoint provided by authors


_TOPK = args.topk
_SCORE_TRESH = args.score_tresh

outputs = []
outputs_raw = []
predictor = DefaultPredictor(cfg)

with torch.no_grad():
    print("START PREDICTION")
    #Providing prediction for single and multiple images
    batchsize = 10
    #Percent of predictions not used
    notused = []
    #Number of images with no predictions
    nopreds = []

    if len(image_paths) < 10:
        batchsize = len(image_paths)
    for i in range(0, len(image_paths), batchsize):
    #for i,img_path in enumerate(image_paths):
        inputs = []
        for img_path in image_paths[i:i+batchsize]:
            img = cv2.imread(img_path)
            #height, width = img.shape[:2]
            #img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
            #image_descriptor = {"image": img, "height": height, "width": width}
            inputs.append(img)
        
        preds = predictor(inputs)
        
        for img_path,pred in zip(image_paths[i:i+batchsize],preds):
            #print("Predict images: ", image_paths)
            image_name = os.path.splitext(os.path.basename(img_path))[0]
            """if args.transformid:    #deprecated
                content_id = image_name.split('_')[0]
                style_id = image_name.split('_')[1]
                image_id = int("%s%s"%(content_id,style_id))
            else:"""
            image_id = image_name #allow string image id

            c = 0
            added = False
            for bbox, keypoints ,score in zip(pred["instances"].pred_boxes, pred["instances"].pred_keypoints, pred["instances"].scores.cpu().numpy()):
                if score.astype("float") >= _SCORE_TRESH:
                    outputs.append({'image_id': image_id, 'image_size': pred["instances"]._image_size,  "category_id": 1, "bbox": bbox.tolist(), "keypoints":keypoints.flatten().tolist(), "score": score.astype("float")})
                    added = True
                if c >= _TOPK - 1:
                    break
                c = c + 1

            if len(pred["instances"]) != 0:
                notused.append(1-c/(len(pred["instances"])))
            if added is False:
                nopreds.append(img_path)

            outputs_raw.append(pred)

        if i%100 == 0 and i!=0:
            print("Processed %d images."%(i+batchsize))
    print("PREDICTION FINISHED")
    print("Percentage of not used predictions (averaged): ", np.mean(notused))
    print("Number of images with no predictions: ", len(nopreds))
    print("Number of images with predictions: ", len(image_paths) - len(nopreds))

    print("Mean number of poses per image: ", len(outputs)/(len(image_paths) - len(nopreds)))

# We can use `Visualizer` to draw the predictions on the image.
#print(MetadataCatalog.get("my_dataset_val"))
MetadataCatalog.get("my_dataset_val").set(keypoint_names=COCO_PERSON_KEYPOINT_NAMES,
                                          keypoint_flip_map=COCO_PERSON_KEYPOINT_FLIP_MAP,
                                          keypoint_connection_rules=KEYPOINT_CONNECTION_RULES) 

#Specification of a threshold for the keypoints in: /home/althausc/.local/lib/python3.6/site-packages/detectron2/utils/visualizer.py


output_dir = os.path.join('/home/althausc/master_thesis_impl/detectron2/out/art_predictions', args.target)
output_dir = os.path.join(output_dir, datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
else:
    raise ValueError("Output directory %s already exists."%output_dir)


#output format of keypoints: (x, y, v), v indicates visibility— v=0: not labeled (in which case x=y=0), v=1: labeled but not visible, and v=2: labeled and visible  
with open(os.path.join(output_dir,"maskrcnn_predictions.json"), 'w') as f:
    json.dump(outputs, f, separators=(', ', ': '))

if args.target == 'train':
    #Writing config to file
    with open(os.path.join(output_dir, 'config.txt'), 'a') as f:
        f.write("Src Image Folder: %s"%(args.image_folder if args.image_folder is not None else args.image_path) + os.linesep)
        f.write("Topk: %d"%args.topk + os.linesep)
        f.write("Score Treshold: %f"%args.score_tresh + os.linesep)
        f.write("Number of images: %d"%len(image_paths) + os.linesep)


def visualize_and_save(img_path, output_dir, preds, args, mode, topk=None):
    if mode == 'all':
        #Draw unfiltered predictions
        v = Visualizer(cv2.imread(img_path)[:, :, ::-1],MetadataCatalog.get("my_dataset_val"), scale=1.2)
        out = v.draw_instance_predictions(preds["instances"].to("cpu"), args.vistresh)
        img_name = os.path.basename(img_path)
        if out == None:
            print("Warning: Image is none.")
        cv2.imwrite(os.path.join(output_dir, img_name),out.get_image()[:, :, ::-1])
        

    elif mode == 'topk':
        #Draw topk predictions
        v = Visualizer(cv2.imread(img_path)[:, :, ::-1],MetadataCatalog.get("my_dataset_val"), scale=1.2)
    
        obj = Instances(image_size=preds["imagesize"])
        obj.set('scores', torch.Tensor(preds["scores"]))
        obj.set('pred_boxes', Boxes(torch.Tensor(preds["bboxes"])))
        obj.set('pred_keypoints', torch.Tensor(preds["keypoints"]))
        
        out = v.draw_instance_predictions(obj, args.vistresh)
        basenames = os.path.splitext(os.path.basename(img_path))
        img_name = os.path.join("%s_topk%s"%(basenames[0], basenames[1]))
        if out == None:
            print("Warning: Image is none.")
        cv2.imwrite(os.path.join(output_dir, img_name),out.get_image()[:, :, ::-1])
        


def get_combined_predictions(singlepreds):
     #Zip single predictions (every annotation is one item) to composed prediction (like output of model)
    singlepreds = sorted(singlepreds, key=lambda x: x['image_id'])
    
    if args.image_folder is None:
        imgdir = os.path.dirname(args.image_path)
    else:
        imgdir = args.image_folder 

    grouped = {}
    for pred_entry in singlepreds:
        imagepath = os.path.join(imgdir, "%s.jpg"%pred_entry['image_id'])
        if imagepath not in grouped:
            grouped[imagepath] = [pred_entry]
        else:
            grouped[imagepath].append(pred_entry)
   
    combined = []
    for img_path, pred_items in grouped.items():
        preds = {}
        preds['imagepath'] = img_path
        preds['imagesize'] = pred_items[0]['image_size']
        preds['keypoints'] = [ [p['keypoints'][i:i+3] for i in range(0, len(p['keypoints']), 3)] for p in pred_items]
        preds['scores'] = [ p['score'] for p in pred_items]
        preds['bboxes'] = [ p['bbox'] for p in pred_items]
        combined.append(preds)
    return combined

if args.vis:
    print("Visualize the predictions onto the original image(s) ...")
    visability_means = []
    print("Draw all/unfiltered predictions...")
    
    for img_path, pred_out in zip(image_paths, outputs_raw):
        visualize_and_save(img_path, output_dir, pred_out, args, 'all')
        
        #Debugging
        for kpt_list in pred_out["instances"].pred_keypoints.cpu():
            kpt_list = kpt_list.numpy()
            visability_means.append(np.sum(kpt_list[:,2])/len(kpt_list))
        #Debugging end
    
    print("Visabilitiy score stats:")
    #print("Mean: ", visability_means)
    Q1, median, Q3 = np.percentile(visability_means, [25, 50, 75])
    print("Min: {} , Q1: {}, Median: {}, Q3: {}, Max: {}".format(min(visability_means), Q1, median, Q3 ,max(visability_means)))
    print("Draw all/unfiltered predictions done.")

    print("Draw topk + treshold predictions...")
    for preds in get_combined_predictions(outputs):
        visualize_and_save(preds['imagepath'], output_dir, preds, args, 'topk', topk=_TOPK)
    print("Draw topk + treshold predictions done.")

    print("Visualize done.")

if args.visrandom:
    print("Random visualization for validation purposes ...")
    preds_comb = get_combined_predictions(outputs)
    for i in range(100):
        k = random.choice(range(len(image_paths)))
        img_path = image_paths[k]
        pred_out = outputs_raw[k]

        #Draw unfiltered predictions
        visualize_and_save(img_path, output_dir, pred_out, args, 'all')
        #Draw topk predictions
        if any(x['imagepath'] == img_path for x in preds_comb):
            pred_searched = next(item for item in preds_comb if item["imagepath"] == img_path)
            visualize_and_save(img_path, output_dir, pred_searched, args, 'topk', topk=_TOPK)

    print("Random visualization done.")
#Getting categories names & ids
#coco = COCO('/home/althausc/nfs/data/coco_17/annotations/instances_val2017.json')
#print(coco.cats)
print("Output directory: ",output_dir)
