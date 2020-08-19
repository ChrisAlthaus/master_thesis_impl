from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from google.colab.patches import cv2_imshow
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.builtin_meta import KEYPOINT_CONNECTION_RULES, COCO_PERSON_KEYPOINT_NAMES, COCO_PERSON_KEYPOINT_FLIP_MAP
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
parser.add_argument('-image_folder','-imgDir', 
                    help='Path to a directory containing images only.')
parser.add_argument('-vis','-visualize', action='store_true',
                    help='Specify to visualize predictions on images & save.')
parser.add_argument('-visrandom','-validate', action='store_true',
                    help='Specify to randomy visualize k predictions.')

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

cfg = get_cfg()

#cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))
#cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.DEVICE='cpu'
#model = build_model(cfg) 
#DetectionCheckpointer(model).load(args.model_cp) 
image_paths = None
if args.image_folder is not None:
    image_paths = [os.path.join(args.image_folder, x) for x in os.listdir(args.image_folder)]
elif args.image_path is not None:
    image_paths = args.image_path


#output format of keypoints: (x, y, v), v indicates visibility— v=0: not labeled (in which case x=y=0), v=1: labeled but not visible, and v=2: labeled and visible  
from detectron2.data import detection_utils as utils
from detectron2.data.datasets.coco import load_coco_json
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.samplers import InferenceSampler
import itertools, copy
from detectron2.data.build import build_detection_test_loader

from detectron2.data.datasets import register_coco_instances

#Register to DatasetCatalog and MetadataCatalog
register_coco_instances("my_dataset_val", {},"/home/althausc/nfs/data/coco_17/annotations/person_keypoints_val2017.json", "/home/althausc/nfs/data/coco_17/val2017")


model = build_model(cfg) 
model.eval()

"""filter = ["008629",
"050326",
"184762",
"220732",
"252216",
"348881",
"367386",
"434204",
"436551",
"450439",
"498463"]"""
"""
filter = [
"002157",
"027186",
"050331",
"074209",
"074457",
"122962",
"229858",
"231822",
"290293",
"441247",
"548780"
]

im_path_filtered = []
for path in image_paths:
    for f in filter:
        if path.find(f) != -1:
            im_path_filtered.append(path)
            break
print(im_path_filtered)"""


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")) 
#cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.DEVICE='cuda'
cfg.MIN_SIZE_TRAIN= 512
cfg.MODEL.WEIGHTS = args.model_cp

outputs = []
outputs_raw = []
predictor = DefaultPredictor(cfg)

with torch.no_grad():
    print("START PREDICTION")
    batchsize = 10
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

            image_name = os.path.splitext(os.path.basename(img_path))[0]
            content_id = image_name.split('_')[0]
            style_id = image_name.split('_')[1]
            image_id = int("%s%s"%(content_id,style_id))

        
            for bbox, keypoints ,score in zip(pred["instances"].pred_boxes, pred["instances"].pred_keypoints, pred["instances"].scores.cpu().numpy()):
                outputs.append({'image_id': image_id, "category_id": 1, "bbox": bbox.tolist(), "keypoints":keypoints.flatten().tolist(), "score": score.astype("float")})
      
            outputs_raw.append( pred )

        if i%100 == 0 and i!=0:
            print("Processed %d images."%(i+batchsize))
    print("PREDICTION FINISHED")

print("OUTPUT PREDICTIONS:")

print("Size of all output predictions: ", len(outputs))
#print(cfg)

# We can use `Visualizer` to draw the predictions on the image.
#print(MetadataCatalog.get("my_dataset_val"))
MetadataCatalog.get("my_dataset_val").set(keypoint_names=COCO_PERSON_KEYPOINT_NAMES,
                                          keypoint_flip_map=COCO_PERSON_KEYPOINT_FLIP_MAP,
                                          keypoint_connection_rules=KEYPOINT_CONNECTION_RULES) 

#Specification of a threshold for the keypoints in: /home/althausc/.local/lib/python3.6/site-packages/detectron2/utils/visualizer.py


output_dir = os.path.join('/home/althausc/master_thesis_impl/detectron2/out/art_predictions', datetime.datetime.now().strftime('%m/%d_%H-%M-%S'))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
else:
    raise ValueError("Output directory %s already exists."%output_dir)

with open(os.path.join(output_dir,"result.json"), 'w') as f:
    json.dump(outputs, f, separators=(', ', ': '))

if args.vis:
    for img_path, pred_out in zip(image_paths, outputs_raw):
        v = Visualizer(cv2.imread(img_path)[:, :, ::-1],MetadataCatalog.get("my_dataset_val"), scale=1.2)
        out = v.draw_instance_predictions(pred_out["instances"].to("cpu"))
        img_name = os.path.basename(img_path)
        if out == None:
            print("img is none")
        cv2.imwrite(os.path.join(output_dir, img_name),out.get_image()[:, :, ::-1])

if args.visrandom:
    for i in range(100):
        k = random.choice(range(len(image_paths)))
        img_path = image_paths[k]
        pred_out = outputs_raw[k]
        v = Visualizer(cv2.imread(img_path)[:, :, ::-1],MetadataCatalog.get("my_dataset_val"), scale=1.2)
        out = v.draw_instance_predictions(pred_out["instances"].to("cpu"))
        img_name = os.path.basename(img_path)
        if out == None:
            print("img is none")
        cv2.imwrite(os.path.join(output_dir, img_name),out.get_image()[:, :, ::-1])

#Getting categories names & ids
#coco = COCO('/home/althausc/nfs/data/coco_17/annotations/instances_val2017.json')
#print(coco.cats)
