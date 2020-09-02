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
import datetime
import csv

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
#from DefaultPredictor import DefaultPredictor 
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
import detectron2.data.build as build
from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from CustomTrainers import *

from plotAveragePrecisions import plotAPS

from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import torch

import argparse


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



parser = argparse.ArgumentParser()
parser.add_argument('-finetune','-fL',required=True, 
                    help='Specify which layers should be trained. Either RESNET, HEADSALL or ALL.')
parser.add_argument('-resume','-checkpoint', 
                    help='Train model from checkpoint given by path.')
parser.add_argument('-numepochs','-epochs', type=int, help='Number of epochs to train.')

args = parser.parse_args()

if args.finetune not in ["RESNETF", "RESNETL", "HEADSALL", "ALL",'EVALBASELINE','FPN+HEADS','SCRATCH']:
    raise ValueError("Not specified a valid training mode for layers.")
if args.resume is not None:
    if not os.path.isfile(args.resume):
        raise ValueError("Checkpoint does not exists.")


#im = cv2.imread("/home/althausc/nfs/data/coco_17_small/train2017_styletransfer/000000000260_049649.jpg")

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
#cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
#cfg.MODEL.DEVICE='cpu'
cfg.MODEL.DEVICE='cuda'


# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml")
if args.finetune != 'SCRATCH':
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
else:
    cfg.MODEL.WEIGHTS = ""


#predictor = DefaultPredictor(cfg)
#outputs = predictor(im)

# We can use `Visualizer` to draw the predictions on the image.
#v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
#out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#cv2_imshow(out.get_image()[:, :, ::-1])
#cv2.imwrite("output.jpg",out.get_image()[:, :, ::-1])


#TRAIN ON STYLE_TRANSFERED DATASET
from detectron2.data.datasets import register_coco_instances

#Register to DatasetCatalog and MetadataCatalog
register_coco_instances("my_dataset_train", {},"/home/althausc/nfs/data/coco_17_medium/annotations_styletransfer/person_keypoints_train2017_stAPI.json", "/home/althausc/nfs/data/coco_17_medium/train2017_styletransfer")

register_coco_instances("my_dataset_val", {}, "/home/althausc/nfs/data/coco_17_medium/annotations_styletransfer/person_keypoints_val2017_stAPI.json", "/home/althausc/nfs/data/coco_17_medium/val2017_styletransfer")
#register_coco_instances("my_dataset_val", {},"/home/althausc/nfs/data/coco_17/annotations/person_keypoints_val2017.json", "/home/althausc/nfs/data/coco_17/val2017")
#register_coco_instances("my_dataset_val", {}, "/home/althausc/nfs/data/coco_17_small/annotations_styletransfer/person_keypoints_val2017_stAPI.json", "/home/althausc/nfs/data/coco_17_small/val2017_styletransfer")

#print(MetadataCatalog.get("my_dataset_train"))
metadata = MetadataCatalog.get("my_dataset_train").set(keypoint_names=COCO_PERSON_KEYPOINT_NAMES, keypoint_flip_map=COCO_PERSON_KEYPOINT_FLIP_MAP) 
metadata = MetadataCatalog.get("my_dataset_train")


dataset = DatasetCatalog.get("my_dataset_train")


cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)     #actually flag is not test but validation set
cfg.DATALOADER.NUM_WORKERS = 2 # Number of data loading threads

cfg.INPUT.MIN_SIZE_TRAIN = 512#(640, 672, 704, 736, 768, 800) #512  #Size of the smallest side of the image during training
    	                     #Defaults: (640, 672, 704, 736, 768, 800)
#cfg.update({'INPUT.FLIP.PROBABILITY': 0.25})
#cfg.INPUT.FLIP.PROBABILITY = 0.25
cfg.INPUT.CROP.TYPE = "relative_range"
cfg.INPUT.CROP.SIZE = [0.9, 0.9]

#Training Parameters
cfg.SOLVER.IMS_PER_BATCH = 4 # Number of images per batch across all machines.

num_epochs = args.numepochs   
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 # 2 faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon) ?
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 17
cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE = 1 #10 # Images with too few (or no) keypoints are excluded from training (default: 1)

output_dir = os.path.join('/home/althausc/master_thesis_impl/detectron2/out/checkpoints', datetime.datetime.now().strftime('%m/%d_%H-%M-%S_'+args.finetune.lower()))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
else:
    raise ValueError("Output directory %s for checkpoints already exists. Please wait a few minutes."%output_dir)

cfg.OUTPUT_DIR = output_dir



def get_iterations_for_epochs(dataset, num_epochs, batch_size):
    """Computes the exact number of iterations for n epochs.
       Since the trainer later will filter out images respective to number of keypoints & is_crowded,
       the same step has to be done here to exactly intefer the number of later used images.""" 
    print("Images in datasets before removing images: ",len(dataset))
    dataset= build.filter_images_with_only_crowd_annotations(dataset)
    dataset= build.filter_images_with_few_keypoints(dataset, cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE)
    print("Images in datasets after removing images: ",len(dataset))
    print(len(dataset), batch_size)
    
    one_epoch = len(dataset) / batch_size
    max_iter = int(one_epoch * num_epochs)
    print("Max iterations: ",max_iter, "1 epoch: ",one_epoch)
    #time.sleep(2)
    return max_iter,one_epoch

if args.finetune != 'EVALBASELINE':
    max_iter, epoch_iter = get_iterations_for_epochs(dataset, num_epochs, cfg.SOLVER.IMS_PER_BATCH)
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.TEST.EVAL_PERIOD = int(epoch_iter/2)   #Evaluation once at the end of each epoch, Set to 0 to disable.
    cfg.TEST.PLOT_PERIOD = int(epoch_iter) # Plot val & train loss curves at every second iteration 
                                            # and save as image in checkpoint folder. Disable: -1
else:
    max_iter = 100
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.TEST.EVAL_PERIOD = 50
    cfg.TEST.PLOT_PERIOD = 200

if args.finetune == 'ALL' or args.finetune == 'SCRATCH':                                            
    cfg.MODEL.RESNETS.NORM = "BN" 
cfg.SOLVER.BASE_LR = 0.001#0.0025  # pick a good LR   #TODO: different values
cfg.SOLVER.GAMMA = 0.95

steps_exp = np.linspace(0,1,12)[1:-1] * max_iter
steps_exp = np.linspace(191893/max_iter,1,12)[0:-1] * max_iter

cfg.SOLVER.STEPS = tuple(steps_exp)
#(int(5/10*max_iter),int(6/10*max_iter), int(7/10*max_iter),int(8/10*max_iter),int(9/10*max_iter),int(98/100*max_iter))
#(int(7/9*max_iter), int(8/9*max_iter))#(int(55/90*max_iter),int(75/90*max_iter),int(85/90*max_iter))#(int(7/9*max_iter), int(8/9*max_iter)) # The iteration marks to decrease learning rate by GAMMA.                                          
#TODO: dump config file for prediction

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
#trainer = DefaultTrainer(cfg) 
trainer = COCOTrainer(cfg) #"multigpu"
#trainer.build_evaluator(cfg, "my_dataset_val") #not necessary?!

#TRIED TO SAVE MODEL GRAPH VISUALIZATION
#model = trainer.model
#shape_of_first_layer = list(model.parameters())[0].shape
#print("Shape of first layer: ",shape_of_first_layer)

#writer = SummaryWriter(cfg.OUTPUT_DIR)
#x = [{'image': torch.randn(3,800,1190), 'height': 336, 'width': 500}]
#x = torch.randn(3,800,1190,1,1)
#x = torch.randn(3,800,1190)

#y = predictor.model(x)
#writer.add_graph(predictor.model,x)
#writer.close()
#summary(predictor.model, x)
#torch.save(trainer.model, os.path.join(cfg.OUTPUT_DIR,'rcnn_model.pth'))

model = trainer.model
from detectron2.layers.wrappers import BatchNorm2d
from detectron2.layers.batch_norm import FrozenBatchNorm2d

def BNFrozentoBN(model, layer_ids, inverse=False):

    for block_name,l_ids in layer_ids.items():
        blocks = getattr(model.backbone.bottom_up, block_name)
        if isinstance(l_ids,list):
            blocks = blocks[l_ids[0]:l_ids[1]+1]
        if block_name == 'stem':
            blocks = [blocks]

        for block in blocks:
            for l_name in ['shortcut','conv1','conv2','conv3']:
                try:
                    layer = getattr(block, l_name)
                    weights = layer.norm.weight
                    num_features = layer.norm.num_features
                    if not inverse:
                        layer.norm = BatchNorm2d(num_features)
                        layer.norm.weights = weights
                        print("Changed layer %s to batched norm unfreezed."%('model.backbone.bottom_up.res2.'+l_name))
                    else:
                        if not isinstance(layer.norm, FrozenBatchNorm2d):
                            layer.norm = FrozenBatchNorm2d(num_features)
                            layer.norm.weights = weights

                except AttributeError:
                    print("Debug: No valid layer: %s."%('model.backbone.bottom_up.res2.'+l_name))
    model.cuda()


print("Save model's state_dict:")
layer_to_params_str = ""
with open(os.path.join(cfg.OUTPUT_DIR, 'layer_params_overview.txt'), 'w') as f:
    for param_tensor in model.state_dict():
        line = "{} \t {}".format(param_tensor, model.state_dict()[param_tensor].size())
        f.write(line + os.linesep)

print("Save model's architecture:")
with open(os.path.join(cfg.OUTPUT_DIR, 'model_architectur.txt'), 'w') as f:
    print(list(model.children()),file=f)

print("Save model's configuration:")
with open(os.path.join(cfg.OUTPUT_DIR, 'model_conf.txt'), 'w') as f:
    print(cfg,file=f)

#print(list(len(model.named_parameters())))

#FPN = list(model.children())[0]
#ResNet = FPN.bottom_up
#for s in ResNet.stages_and_names:
#   if s[1] == 'res4':                                             
layername_prefixes = {"ResNet":"backbone.bottom_up", "RPN_ANCHOR":"proposal_generator.anchor_generator", "RPN_HEAD":"proposal_generator.rpn_head",
                      "ROI_HEAD":"roi_heads.box_head", "ROI_PREDICTOR":"roi_heads.box_predictor", "ROI_KEYPOINT":"roi_heads.keypoint_head",
                      "FPN":"backbone.fpn"} 
trainlayers = []
#for l,w in list(model.named_parameters()):
#    print(l)

#Mapping for layers to enable batch normalization
layersbn= { "ALL": {'stem':'', 'res2':''},
            "SCRATCH": {'stem':'', 'res2':''},
            "RESNETL": {'res4':[2,5], 'res5':[0,2]},
            "RESNETF": {'res2':[0,2], 'res3':[0,3]},
            "HEADSALL": ["ROI_KEYPOINT","ROI_PREDICTOR","ROI_HEAD","RPN_HEAD","RPN_ANCHOR"]}

if args.finetune == "ALL" or args.finetune == 'SCRATCH':
    #Replace FronzenBatchedNorm2D with BatchedNorm2D in the first two blocks
    #because not implemented by cfg file
    BNFrozentoBN(model, layersbn['ALL'])
    print("test")

elif args.finetune == "RESNETL":
    #layernamesResNet = ['res1','res2','res3','res4','res5']
    trainConvBlocks = layersbn['RESNETL'] #[start,end] with end inclusive

    layerNoFreeze = []
    #Adding resnet layers which should be trained/ not freezed
    for l_name,indices in trainConvBlocks.items():
        layerNoFreeze.extend(["backbone.bottom_up.{}.{}".format(l_name,i) for i in range(indices[0],indices[1]+1)])
    print("Layers to train: ",layerNoFreeze)

    for name, param in list(model.named_parameters()):
        isTrainLayer = any([name.find(l_name) != -1 for l_name in layerNoFreeze])
        print("name:",name)
        if not isTrainLayer:
            param.requires_grad = False     #freeze layer
            print("freeze")
        else:
            print("Train layer: ",name)
            trainlayers.append(name)

    #Unfreeze batch normalization layers
    BNFrozentoBN(model, trainConvBlocks)

elif args.finetune == "RESNETF":
    #layernamesResNet = ['res1','res2','res3','res4','res5']
    trainConvBlocks = layersbn['RESNETF'] #[start,end] with end inclusive

    layerNoFreeze = []
    #Adding resnet layers which should be trained/ not freezed
    for l_name,indices in trainConvBlocks.items():
        layerNoFreeze.extend(["backbone.bottom_up.{}.{}".format(l_name,i) for i in range(indices[0],indices[1]+1)])
    print("Layers to train: ",layerNoFreeze)

    for name, param in list(model.named_parameters()):
        isTrainLayer = any([name.find(l_name) != -1 for l_name in layerNoFreeze])
        if not isTrainLayer:
            param.requires_grad = False     #freeze layer
        else:
            print("Train layer: ",name)
            trainlayers.append(name)

    #Unfreeze batch normalization layers
    BNFrozentoBN(model, trainConvBlocks)

elif args.finetune == 'HEADSALL':
    layersNoFreezePrefix = [layername_prefixes[x] for x in layersbn['HEADSALL']]
    
    for name, param in list(model.named_parameters()):
        isTrainLayer = any([name.find(l_name) != -1 for l_name in layersNoFreezePrefix])
        if not isTrainLayer:
            print("Not training layer: ",name)
            param.requires_grad = False     #freeze layer
        else:
            print("Train layer: ",name)
            trainlayers.append(name)

"""elif args.finetune == 'FPN+HEADS':
    layersNoFreezePrefix = [layername_prefixes[x] for x in ["FPN","ROI_KEYPOINT","ROI_PREDICTOR","ROI_HEAD","RPN_HEAD","RPN_ANCHOR"]]
    
    for name, param in list(model.named_parameters()):
        isTrainLayer = any([name.find(l_name) != -1 for l_name in layersNoFreezePrefix])
        if not isTrainLayer:
            print("Not training layer: ",name)
            param.requires_grad = False     #freeze layer
        else:
            print("Train layer: ",name)
            trainlayers.append(name)"""

elif args.finetune == 'EVALBASELINE':
    for name, param in list(model.named_parameters()):
        if name != 'roi_heads.keypoint_head.score_lowres.weight':
            param.requires_grad = False     #freeze layer
    
    
def save_modelconfigs(outdir, cfg, layersbn_map, args):
    filename = 'run_configs.csv'
    filepath = os.path.join(outdir, filename)

    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            headers = ['Folder', 'NET', 'BN', 'LR', 'Gamma', 'Steps', 'Epochs', 'Data Augmentation', 'Min Keypoints', 'MinSize Train', 'ImPerBatch', 'Additional']
            writer.writerow(headers)
    folder = os.path.basename(cfg.OUTPUT_DIR)
    bnlayers = layersbn_map[args.finetune]
    data_augm = [cfg.INPUT.CROP.SIZE, .. , ..]  #TODO: add additional items to cfg object
    
    row = [folder, args.finetune, bnlayers, cfg.SOLVER.BASE_LR, cfg.SOLVER.GAMMA, cfg.SOLVER.STEPS
            args.numepochs, data_augm, cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE, cfg.INPUT.MIN_SIZE_TRAIN, cfg.SOLVER.IMS_PER_BATCH ]
    with open(filepath, 'a') as f:
        writer.writerow(row)


checkParams = dict()    #Dict to save intial parameters for some random freezed layers
                        #for later checking if really not trained            
numLayersForCheck = 20
numAdded = 0
while(numAdded < 20):
    name, param = random.choice(list(model.named_parameters()))
    #print(name,param)
    if name in trainlayers or name in checkParams.keys():
        continue
    else:
        checkParams.update({name:param})
        numAdded = numAdded + 1

if args.resume is not None:
    print("Resuming training from checkpoint %s."%args.resume)
    DetectionCheckpointer(trainer.model).load(args.resume)
print("resume or load")
trainer.resume_or_load(resume=False)
print("START TRAINING")
trainer.train()
print("TRAINING DONE.")


#Check if some freezed layers have the same weights as before training
for name, param in list(model.named_parameters()):
    if name in checkParams:
        if not torch.equal(param.data, checkParams[name].data):
            print("Layer %s has been modified whereas it should actually been freezed."%name)


#Plot average precision plots
plotAPS(cfg.OUTPUT_DIR)