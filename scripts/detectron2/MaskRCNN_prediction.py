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

from detectron2.data.datasets import register_coco_instances
import torch

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
parser.add_argument('-vis','-visualize', action='store_true',
                    help='Specify to visualize predictions on images & save.')
parser.add_argument('-visrandom','-validate', action='store_true',
                    help='Specify to randomy visualize k predictions.')
parser.add_argument('-vistresh', type=float, default=0.0,
                    help='Specify a treshold for visualization.')
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
cfg.MODEL.DEVICE='cpu'
cfg.MIN_SIZE_TRAIN= 512
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml") 
cfg.MODEL.WEIGHTS = args.model_cp #uncomment for specified checkpoint

outputs = []
outputs_raw = []
predictor = DefaultPredictor(cfg)

with torch.no_grad():
    print("START PREDICTION")
    #Providing prediction for single and multiple images
    batchsize = 10
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
            print("Predict images: ", image_paths)
            image_name = os.path.splitext(os.path.basename(img_path))[0]
            if args.transformid:
                content_id = image_name.split('_')[0]
                style_id = image_name.split('_')[1]
                image_id = int("%s%s"%(content_id,style_id))
            else:
                image_id = image_name #allow string image id

        
            for bbox, keypoints ,score in zip(pred["instances"].pred_boxes, pred["instances"].pred_keypoints, pred["instances"].scores.cpu().numpy()):
                outputs.append({'image_id': image_id, "category_id": 1, "bbox": bbox.tolist(), "keypoints":keypoints.flatten().tolist(), "score": score.astype("float")})
      
            outputs_raw.append( pred )

        if i%100 == 0 and i!=0:
            print("Processed %d images."%(i+batchsize))
    print("PREDICTION FINISHED")

print("OUTPUT PREDICTIONS:")

print("Size of all output predictions: ", len(outputs))

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

if args.vis:
    print("Visualize the predictions onto the original image(s) ...")
    for img_path, pred_out in zip(image_paths, outputs_raw):
        v = Visualizer(cv2.imread(img_path)[:, :, ::-1],MetadataCatalog.get("my_dataset_val"), scale=1.2)
        out = v.draw_instance_predictions(pred_out["instances"].to("cpu"), args.vistresh)
        img_name = os.path.basename(img_path)
        if out == None:
            print("img is none")
        cv2.imwrite(os.path.join(output_dir, img_name),out.get_image()[:, :, ::-1])
    print("Visualize done.")

if args.visrandom:
    print("Random visualization for validation purposes ...")
    for i in range(100):
        k = random.choice(range(len(image_paths)))
        img_path = image_paths[k]
        pred_out = outputs_raw[k]
        v = Visualizer(cv2.imread(img_path)[:, :, ::-1],MetadataCatalog.get("my_dataset_val"), scale=1.2)
        out = v.draw_instance_predictions(pred_out["instances"].to("cpu"), args.vistresh)
        img_name = os.path.basename(img_path)
        if out == None:
            print("img is none")
        cv2.imwrite(os.path.join(output_dir, img_name),out.get_image()[:, :, ::-1])
    print("Random visualization done.")
#Getting categories names & ids
#coco = COCO('/home/althausc/nfs/data/coco_17/annotations/instances_val2017.json')
#print(coco.cats)
print("Output directory: ",output_dir)
