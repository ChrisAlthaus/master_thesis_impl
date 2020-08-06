from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from google.colab.patches import cv2_imshow
from detectron2.data import MetadataCatalog

import argparse
import os

from pycocotools.coco import COCO
import cv2
import datetime


parser = argparse.ArgumentParser()
parser.add_argument('-model_cp','-mc',required=True, 
                    help='Path to the model checkpoint file.')
parser.add_argument('-image_path','-img', 
                    help='Path to the image for which pose inference will be calculated.')
parser.add_argument('-image_folder','-imgDir', 
                    help='Path to a directory containing images only.')

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
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.DEVICE='cpu'
#model = build_model(cfg) 
#DetectionCheckpointer(model).load(args.model_cp) 
image_paths = None
if args.image_folder is not None:
    image_paths = [os.path.join(args.image_folder, x) for x in os.listdir(args.image_folder)]
elif args.image_path is not None:
    image_paths = args.image_path

outputs = []
"""
pred = DefaultPredictor(cfg)
for img_path in image_paths:
    inputs = cv2.imread(img_path)
    outputs.append( pred(inputs) )"""
#output format of keypoints: (x, y, v), v indicates visibility— v=0: not labeled (in which case x=y=0), v=1: labeled but not visible, and v=2: labeled and visible  

images = []
for img_path in image_paths:
    images.append(cv2.imread(img_path))

model = build_model(cfg) 
model.eval()
with torch.no_grad():
  outputs = model([images])
print(outputs)
exit(1)
#print(outputs.shape)
#print(cfg)

# We can use `Visualizer` to draw the predictions on the image.
print(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
#Specification of a threshold for the keypoints in: /home/althausc/.local/lib/python3.6/site-packages/detectron2/utils/visualizer.py
v = Visualizer(inputs[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)

output_dir = os.path.join('/home/althausc/master_thesis_impl/detectron2/out/art_predictions', datetime.datetime.now().strftime('%m/%d_%H-%M-%S'))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
else:
    raise ValueError("Output directory %s already exists."%output_dir)

for img_path, pred_out in zip(image_paths, outputs):
    out = v.draw_instance_predictions(pred_out["instances"].to("cpu"))
    img_name = os.path.basename(img_path)

    cv2.imwrite(os.path.join(output_dir, img_name),out.get_image()[:, :, ::-1])

#Getting categories names & ids
coco = COCO('/home/althausc/nfs/data/coco_17/annotations/instances_val2017.json')
#print(coco.cats)
