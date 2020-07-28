# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import urllib
from google.colab.patches import cv2_imshow

import time

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
import detectron2.data.build as build
from CustomTrainers import *

from torchsummary import summary


COCO_PERSON_KEYPOINT_NAMES = (
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)
# fmt: on

# Pairs of keypoints that should be exchanged under horizontal flipping
COCO_PERSON_KEYPOINT_FLIP_MAP = (
    ("left_eye", "right_eye"),
    ("left_ear", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
)

#im = cv2.imread("/home/althausc/nfs/data/coco_17_small/train2017_styletransfer/000000000260_049649.jpg")

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
#cfg.MODEL.DEVICE='cpu'
cfg.MODEL.DEVICE='cuda'

#print(cfg)

# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml")
#predictor = DefaultPredictor(cfg)
#outputs = predictor(im)

#print(outputs)
#time.sleep(20)


# We can use `Visualizer` to draw the predictions on the image.
#v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
#out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#cv2_imshow(out.get_image()[:, :, ::-1])
#cv2.imwrite("output.jpg",out.get_image()[:, :, ::-1])


#TRAIN ON STYLE_TRANSFERED DATASET
from detectron2.data.datasets import register_coco_instances

#Register to DatasetCatalog and MetadataCatalog
register_coco_instances("my_dataset_train", {}, "/home/althausc/nfs/data/coco_17_small/annotations_styletransfer/person_keypoints_train2017_stNorm.json", "/home/althausc/nfs/data/coco_17_small/train2017_styletransfer")
register_coco_instances("my_dataset_val", {}, "/home/althausc/nfs/data/coco_17_small/annotations_styletransfer/person_keypoints_val2017_stNorm.json", "/home/althausc/nfs/data/coco_17_small/val2017_styletransfer")
#print(MetadataCatalog.get("my_dataset_train"))
metadata = MetadataCatalog.get("my_dataset_train").set(keypoint_names=COCO_PERSON_KEYPOINT_NAMES, keypoint_flip_map=COCO_PERSON_KEYPOINT_FLIP_MAP) 
metadata = MetadataCatalog.get("my_dataset_train")


dataset = DatasetCatalog.get("my_dataset_train")


cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)     #actually flag is not test but validation set
cfg.DATALOADER.NUM_WORKERS = 2

#Training Parameters
cfg.SOLVER.IMS_PER_BATCH = 16
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
num_epochs = 10   
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 2  # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon) ?
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 17
cfg.OUTPUT_DIR = '/home/althausc/master_thesis_impl/detectron2/out/checkpoints'


def get_iterations_for_epochs(dataset, num_epochs, batch_size):
    """Computes the exact number of iterations for n epochs.
       Since the trainer later will filter out images respective to number of keypoints & is_crowded,
       the same step has to be done here to exactly intefer the number of later used images.""" 
    print("Images in datasets before removing images: ",len(dataset))
    dataset= build.filter_images_with_only_crowd_annotations(dataset)
    dataset= build.filter_images_with_few_keypoints(dataset, cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE)
    print("Images in datasets after removing images: ",len(dataset))

    one_epoch = len(dataset) / batch_size
    max_iter = int(one_epoch * num_epochs)
    print("Max iterations: ",max_iter)
    return max_iter,one_epoch

max_iter, epoch_iter = get_iterations_for_epochs(dataset, num_epochs, cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE)
cfg.SOLVER.MAX_ITER = max_iter
cfg.TEST.EVAL_PERIOD = epoch_iter   #evaluation once at the end of each epoch


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
print("11111111111111111111111111")
#trainer = DefaultTrainer(cfg) 
trainer = COCOTrainer(cfg)
model = trainer.model
#summary(model,(256,256,3)) error

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
x = torch.randn(1, 3, 224, 224)
writer.add_graph(model,x)
writer.close()

exit(1)
with open(os.path.join(cfg.OUTPUT_DIR, 'model_architectur.txt'), 'w') as f:
    print(list(model.children()),file=f)

exit(1)
#trainer.build_evaluator(cfg, "my_dataset_val") 

print("2222222222222222222222222")

trainer.resume_or_load(resume=False)
trainer.train()
