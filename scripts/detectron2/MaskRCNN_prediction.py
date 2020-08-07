from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from google.colab.patches import cv2_imshow
from detectron2.data import MetadataCatalog
import torch

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


"""
pred = DefaultPredictor(cfg)
for img_path in image_paths:
    inputs = cv2.imread(img_path)
    outputs.append( pred(inputs) )"""
#output format of keypoints: (x, y, v), v indicates visibility— v=0: not labeled (in which case x=y=0), v=1: labeled but not visible, and v=2: labeled and visible  
from detectron2.data import detection_utils as utils
from detectron2.data.datasets.coco import load_coco_json
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.samplers import InferenceSampler
import itertools, copy
from detectron2.data.build import build_detection_test_loader

"""dataset_dicts = load_coco_json('/home/althausc/nfs/data/coco_17/annotations/person_keypoints_val2017.json', '/home/althausc/nfs/data/coco_17/val2017', 'val2017')
dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))"""
from detectron2.data.datasets import register_coco_instances

#Register to DatasetCatalog and MetadataCatalog
register_coco_instances("my_dataset_val", {},"/home/althausc/nfs/data/coco_17/annotations/person_keypoints_val2017.json", "/home/althausc/nfs/data/coco_17/val2017")

""" 
def mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    
    height, width = image.shape[:2]
    image = torch.from_numpy(image.astype("float32").transpose(2, 0, 1))

    return {"image": image,"height": height, "width": width}

dataloader = build_detection_test_loader(cfg, "my_dataset_val", mapper=mapper)

dataset = DatasetFromList(dataset_dicts)
dataset = MapDataset(dataset, mapper)

sampler = InferenceSampler(len(dataset))
batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)
data_loader = torch.utils.data.DataLoader(
       dataset,
       num_workers=4,
       batch_sampler=batch_sampler,
       collate_fn=trivial_batch_collator,
   )
"""

#for i, batch in enumerate(dataloader):
#    print(i,batch)



"""images = []
for img_path in image_paths:
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
    inputs = {"image": img, "height": height, "width": width}
    images.append(inputs)"""

"""model = build_model(cfg) 
model.eval()
print("Number of images which will be prediced for: ", len(dataloader))
with torch.no_grad():
    print("START PREDICTION")
    for i, batch in enumerate(dataloader):
        print(i,batch)
        outputs = model(batch)
        if i%100 == 0:
            print("Processed %d images."%i)
        if i==1000:
            break
    print("PREDICTION FINISHED")"""  #Assertion error in _keypoints_to_heatmap ?

outputs = []

model = build_model(cfg) 
model.eval()
with torch.no_grad():
    print("START PREDICTION")
    for i,img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        inputs = {"image": img, "height": height, "width": width}
        
        outputs.append( model([inputs])[0] )
        break
        if i%100 == 0 and i!=0:
            print("Processed %d images."%i)
            break
    print("PREDICTION FINISHED")


#print(outputs.shape)
#print(cfg)

# We can use `Visualizer` to draw the predictions on the image.
print(MetadataCatalog.get("my_dataset_val"))
#Specification of a threshold for the keypoints in: /home/althausc/.local/lib/python3.6/site-packages/detectron2/utils/visualizer.py


output_dir = os.path.join('/home/althausc/master_thesis_impl/detectron2/out/art_predictions', datetime.datetime.now().strftime('%m/%d_%H-%M-%S'))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
else:
    raise ValueError("Output directory %s already exists."%output_dir)

for img_path, pred_out in zip(image_paths, outputs):
    v = Visualizer(cv2.imread(img_path)[:, :, ::-1], scale=1.2)
    out = v.draw_instance_predictions(pred_out["instances"].to("cpu"))
    img_name = os.path.basename(img_path)

    cv2.imwrite(os.path.join(output_dir, img_name),out.get_image()[:, :, ::-1])

#Getting categories names & ids
coco = COCO('/home/althausc/nfs/data/coco_17/annotations/instances_val2017.json')
#print(coco.cats)
