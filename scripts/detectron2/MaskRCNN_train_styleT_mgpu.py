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

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
#from DefaultPredictor import DefaultPredictor 
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer, launch
import detectron2.data.build as build

from torch.nn.parallel import DistributedDataParallel
import detectron2.utils.comm as comm

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
parser.add_argument('-nGPUs',required=True, type=int, 
                    help='Numer of GPUs to train on.')
args = parser.parse_args()

if args.finetune not in ["RESNETF", "RESNETL", "HEADSALL", "ALL",'EVALBASELINE','FPN+HEADS']:
    raise ValueError("Not specified a valid training mode for layers.")


#im = cv2.imread("/home/althausc/nfs/data/coco_17_small/train2017_styletransfer/000000000260_049649.jpg")
def train(args, output_dir):
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    #cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    #cfg.MODEL.DEVICE='cpu'
    cfg.MODEL.DEVICE='cuda'


    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")

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

    cfg.INPUT.MIN_SIZE_TRAIN = 512  #Size of the smallest side of the image during training
                                #Defaults: (640, 672, 704, 736, 768, 800)

    #Training Parameters
    cfg.SOLVER.IMS_PER_BATCH = 8 # Number of images per batch across all machines.

    num_epochs = 10   
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 # 2 faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon) ?
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 17
    cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE = 1 # Images with too few (or no) keypoints are excluded from training (default: 1)

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
        time.sleep(2)
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
                                                
    
    cfg.SOLVER.BASE_LR = 0.0001 #0.0025  # pick a good LR   #TODO: different values
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.STEPS = (int(6/9*max_iter), int(8/9*max_iter)) # The iteration marks to decrease learning rate by GAMMA.                                          


    #os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
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

    #if comm.is_main_process():
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
    #comm.synchronize()

    #print(list(len(model.named_parameters())))


    layername_prefixes = {"ResNet":"backbone.bottom_up", "RPN_ANCHOR":"proposal_generator.anchor_generator", "RPN_HEAD":"proposal_generator.rpn_head",
                      "ROI_HEAD":"roi_heads.box_head", "ROI_PREDICTOR":"roi_heads.box_predictor", "ROI_KEYPOINT":"roi_heads.keypoint_head",
                      "FPN":"backbone.fpn"} 
    trainlayers = []

    if args.finetune == "RESNETL":
        layernamesResNet = ['res1','res2','res3','res4','res5']
        trainConvBlocks = {'res4':[19,22], 'res5':[0,2]} #[start,end] with end inclusive

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

    elif args.finetune == "RESNETF":
        layernamesResNet = ['res1','res2','res3','res4','res5']
        trainConvBlocks = {'res2':[0,2], 'res3':[0,3]} #[start,end] with end inclusive

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

    elif args.finetune == 'HEADSALL':
        layersNoFreezePrefix = [layername_prefixes[x] for x in ["ROI_KEYPOINT","ROI_PREDICTOR","ROI_HEAD","RPN_HEAD","RPN_ANCHOR"]]
        
        for name, param in list(model.named_parameters()):
            isTrainLayer = any([name.find(l_name) != -1 for l_name in layersNoFreezePrefix])
            if not isTrainLayer:
                print("Not training layer: ",name)
                param.requires_grad = False     #freeze layer
            else:
                print("Train layer: ",name)
                trainlayers.append(name)

    elif args.finetune == 'FPN+HEADS':
        layersNoFreezePrefix = [layername_prefixes[x] for x in ["FPN","ROI_KEYPOINT","ROI_PREDICTOR","ROI_HEAD","RPN_HEAD","RPN_ANCHOR"]]
        
        for name, param in list(model.named_parameters()):
            isTrainLayer = any([name.find(l_name) != -1 for l_name in layersNoFreezePrefix])
            if not isTrainLayer:
                print("Not training layer: ",name)
                param.requires_grad = False     #freeze layer
            else:
                print("Train layer: ",name)
                trainlayers.append(name)

    elif args.finetune == 'EVALBASELINE':
        for name, param in list(model.named_parameters()):
            if name != 'roi_heads.keypoint_head.score_lowres.weight':
                param.requires_grad = False     #freeze layer
    
        
        


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

    #trainer.resume_or_load(resume=False)
    
    print("BUILD MODEL")

    """distributed = comm.get_world_size() > 1
    if distributed:
        trainer.model = DistributedDataParallel(
            trainer.model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )"""
  
    print("START TRAINING")
    return trainer.train()
    #print("TRAINING DONE.")

    
    #Check if some freezed layers have the same weights as before training
    for name, param in list(model.named_parameters()):
        if name in checkParams:
            if not torch.equal(param.data, checkParams[name].data):
                print("Layer %s has been modified whereas it should actually been freezed."%name)


    #Plot average precision plots
    if comm.is_main_process():
        plotAPS(cfg.OUTPUT_DIR)

if __name__ == "__main__":
    output_dir = os.path.join('/home/althausc/master_thesis_impl/detectron2/out/checkpoints', datetime.datetime.now().strftime('%m/%d_%H-%M-%S_'+args.finetune.lower()))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        raise ValueError("Output directory %s for checkpoints already exists. Please wait a few minutes."%output_dir)

    if args.nGPUs > 0 and args.nGPUs <= 8:
        launch(train, args.nGPUs, num_machines=1, machine_rank=0, dist_url='auto', args=(args,output_dir)) #'tcp://127.0.0.1:58636'
    else:
        raise ValueError("Please specify a valid number of GPUs between 1-8.")