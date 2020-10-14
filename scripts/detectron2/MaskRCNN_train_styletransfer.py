# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import urllib

import time
import datetime
import csv

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.layers.wrappers import BatchNorm2d
from detectron2.layers.batch_norm import FrozenBatchNorm2d
import detectron2.data.build as build
from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from CustomTrainers import *
from detectron2.data.datasets import register_coco_instances

from detectron2.data.datasets.builtin_meta import COCO_PERSON_KEYPOINT_NAMES, COCO_PERSON_KEYPOINT_FLIP_MAP


from plotAveragePrecisions import plotAPS

from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import torch
from decimal import Decimal

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-finetune','-fL',required=True, 
                        help='Specify which layers should be trained. Either RESNET, HEADSALL or ALL.')
    parser.add_argument('-resume','-checkpoint', 
                        help='Train model from checkpoint given by path.')
    parser.add_argument('-numepochs','-epochs', type=int, help='Number of epochs to train.')
    parser.add_argument('-addconfig', help='Add selected configurations as an additional row to a csv file.', action="store_true")
    
    args = parser.parse_args()
    
    if args.finetune not in ["RESNETF", "RESNETL", "HEADSALL", "ALL",'EVALBASELINE','FPN+HEADS','SCRATCH']:
        raise ValueError("Not specified a valid training mode for layers.")
    if args.resume is not None:
        if not os.path.isfile(args.resume):
            raise ValueError("Checkpoint does not exists.")
    
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")) #"COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"
    
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    if args.finetune != 'SCRATCH':
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml") #"COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"
    else:
        cfg.MODEL.WEIGHTS = ""

    cfg.MODEL.DEVICE='cuda' #'cpu'   


    output_dir = os.path.join('/home/althausc/master_thesis_impl/detectron2/out/checkpoints', datetime.datetime.now().strftime('%m-%d_%H-%M-%S_baseline'))#+args.finetune.lower()))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Successfully created output directory: ", output_dir)
    else:
        raise ValueError("Output directory %s for checkpoints already exists. Please wait a few minutes."%output_dir)
    cfg.OUTPUT_DIR = output_dir

    # ---------------------------- DATASETS ------------------------------
    #Register to DatasetCatalog and MetadataCatalog
    train_ann = "/home/althausc/nfs/data/coco_17_medium/annotations_styletransfer/person_keypoints_train2017_stAPI.json"
    train_dir = "/home/althausc/nfs/data/coco_17_medium/train2017_styletransfer"
    val_ann = "/home/althausc/nfs/data/coco_17_medium/annotations_styletransfer/person_keypoints_val2017_stAPI.json"
    val_dir = "/home/althausc/nfs/data/coco_17_medium/val2017_styletransfer"
    
    register_coco_instances("my_dataset_train", {}, train_ann, train_dir)
    register_coco_instances("my_dataset_val", {}, val_ann, val_dir)
    metadata = MetadataCatalog.get("my_dataset_train").set(keypoint_names=COCO_PERSON_KEYPOINT_NAMES, keypoint_flip_map=COCO_PERSON_KEYPOINT_FLIP_MAP) 
    metadata = MetadataCatalog.get("my_dataset_train")
    #Get dataset for calculating steps etc.
    dataset = DatasetCatalog.get("my_dataset_train")
    
    
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)     #actually flag is not test but validation set
    cfg.DATALOADER.NUM_WORKERS = 2 # Number of data loading threads
    
    # ------------------------- DATA AUGMENTATION ----------------------------
    cfg.INPUT.CROP.TYPE = "relative_range"
    cfg.INPUT.CROP.SIZE = [0.9, 0.9]
    cfg.INPUT.CROP.ENABLED = False #True
    
    cfg.DATA_FLIP_PROBABILITY = 0.25 
    cfg.DATA_FLIP_ENABLED = False #True
    cfg.ROTATION = [-15,15]
    cfg.ROTATION_ENABLED = False #True
    

    #cfg.INPUT.MIN_SIZE_TRAIN = 512#(640, 672, 704, 736, 768, 800) #512  #Size of the smallest side of the image during training
        	                     #Defaults: (640, 672, 704, 736, 768, 800)
    
    """
    # ------------------------ SPECIFIC LAYER PARAMETERS -------------------------
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 # 2 faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon) ?
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 17
    cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE = 1 #10 # Images with too few (or no) keypoints are excluded from training (default: 1)
    #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model, only for test?!

    # ---------------------------- SOLVER PARAMETERS ---------------------------
    """
    cfg.SOLVER.IMS_PER_BATCH = 2 #4, original=2 # Number of images per batch across all machines.
    """
    cfg.SOLVER.BASE_LR = 0.001#0.0025, original=0.001  # pick a good LR   #TODO: different values
    cfg.SOLVER.GAMMA = 0.95
    """
    if args.finetune != 'EVALBASELINE':
        max_iter, epoch_iter = get_iterations_for_epochs(dataset, args.numepochs, cfg.SOLVER.IMS_PER_BATCH, cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE)
        #cfg.SOLVER.MAX_ITER = max_iter
        cfg.TEST.EVAL_PERIOD = int(epoch_iter/2)   #Evaluation once at the end of each epoch, Set to 0 to disable.
        cfg.TEST.PLOT_PERIOD = int(epoch_iter) # Plot val & train loss curves at every second iteration 
                                                # and save as image in checkpoint folder. Disable: -1
    else:
        max_iter = 100
        cfg.SOLVER.MAX_ITER = max_iter
        cfg.TEST.EVAL_PERIOD = 50
        cfg.TEST.PLOT_PERIOD = 200
    """
    steps_exp = np.linspace(0,1,12)[1:-1] * max_iter
    #For lr decay from a specific iteration number
    #iternum = 191893
    #steps_exp = np.linspace(iternum/max_iter,1,12)[0:-1] * max_iter
    #(int(5/10*max_iter),int(6/10*max_iter), int(7/10*max_iter),int(8/10*max_iter),int(9/10*max_iter),int(98/100*max_iter))
    #(int(7/9*max_iter), int(8/9*max_iter))#(int(55/90*max_iter),int(75/90*max_iter),int(85/90*max_iter))#(int(7/9*max_iter), int(8/9*max_iter)) # The iteration marks to decrease learning rate by GAMMA.  
    steps_exp.astype(np.int)
    cfg.SOLVER.STEPS = tuple(steps_exp)"""
    
    # ------------------------- APPLY CONFIG & GET MODEL ------------------------
    trainer = COCOTrainer(cfg)    
    model = trainer.model
    
    # -------------------------- BATCH NORMALIZATION SETUP ------------------------
    #if args.finetune == 'ALL' or args.finetune == 'SCRATCH':                                            
    #    cfg.MODEL.RESNETS.NORM = "BN" 
   
    trainlayers = [] #setupLayersAndBN(model, args.finetune)
    
    # ----------------------------- SAVING CONFIGS -------------------------------
    if args.addconfig:
        print("Add reduced config to overview file: ")
        cfgdir = '/home/althausc/master_thesis_impl/detectron2/out/checkpoints'
        save_modelconfigs(cfgdir, cfg, BN_LAYERS, args) 
        
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

    # -------------------- APPLY CFG & CHANGED MODEL ---------------------
  
    trainer.model = model #necessary?
    if args.resume is not None:
        print("Resuming training from checkpoint %s."%args.resume)
        DetectionCheckpointer(trainer.model).load(args.resume)
    
    #trainer.resume_or_load(resume=False)
    # --------------------- TRAIN MODEL ----------------------------------
    checkparams = get_checkparams(model, trainlayers)

    print("START TRAINING")
    trainer.train()
    print("TRAINING DONE.")

    check_modelparams(model, checkparams)

    #Plot average precision plots
    plotAPS(cfg.OUTPUT_DIR)


def get_iterations_for_epochs(dataset, num_epochs, batch_size, min_kpts):
    """Computes the exact number of iterations for n epochs.
       Since the trainer later will filter out images respective to number of keypoints & is_crowded,
       the same step has to be done here to exactly intefer the number of later used images.""" 
    print("Images in datasets before removing images: ",len(dataset))
    dataset= build.filter_images_with_only_crowd_annotations(dataset)
    dataset= build.filter_images_with_few_keypoints(dataset, min_kpts)
    print("Images in datasets after removing images: ",len(dataset))
    
    one_epoch = len(dataset) / batch_size
    max_iter = int(one_epoch * num_epochs)
    print("Max iterations: ",max_iter, "1 epoch: ",one_epoch)
    #time.sleep(2)
    return max_iter,one_epoch

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

def setupLayersAndBN(model, trainmode):
    #Freeze specific layers which should not be trained according to trainmode
    #Additional unfreeze corresponding BN layers

    #Mapping for layers to enable batch normalization
    BN_LAYERS = { "ALL": {'stem':'', 'res2':''},
                "SCRATCH": {'stem':'', 'res2':''},
                "RESNETL": {'res4':[2,5], 'res5':[0,2]},
                "RESNETF": {'res2':[0,2], 'res3':[0,3]},
                "HEADSALL": ["ROI_KEYPOINT","ROI_PREDICTOR","ROI_HEAD","RPN_HEAD","RPN_ANCHOR"]}
    RESNET_LAYERNAMES = ['res1','res2','res3','res4','res5']

    PREFIXES_LAYERNAMES = {"ResNet":"backbone.bottom_up", "RPN_ANCHOR":"proposal_generator.anchor_generator", "RPN_HEAD":"proposal_generator.rpn_head",
                          "ROI_HEAD":"roi_heads.box_head", "ROI_PREDICTOR":"roi_heads.box_predictor", "ROI_KEYPOINT":"roi_heads.keypoint_head",
                          "FPN":"backbone.fpn"} 
    #keep track of layers which will be trained (not for "ALL" and "SCRATCH")                      
    trainlayers = []

    if trainmode == "ALL" or trainmode == 'SCRATCH':
        #Replace FronzenBatchedNorm2D with BatchedNorm2D in the first two blocks
        #because not implemented by cfg file
        BNFrozentoBN(model, BN_LAYERS['ALL'])

    elif trainmode == "RESNETL":
        trainConvBlocks = BN_LAYERS['RESNETL'] #[start,end] with end inclusive
        layerNoFreeze = []
        #Adding resnet layers which should be trained/ not freezed
        for l_name,indices in trainConvBlocks.items():
            layerNoFreeze.extend(["backbone.bottom_up.{}.{}".format(l_name,i) for i in range(indices[0],indices[1]+1)])
        print("Layers to train: ",layerNoFreeze)
        for name, param in list(model.named_parameters()):
            isTrainLayer = any([name.find(l_name) != -1 for l_name in layerNoFreeze])
            if not isTrainLayer:
                param.requires_grad = False     #freeze layer
                print("Freeze layer: ",name)
            else:
                print("Train layer: ",name)
                trainlayers.append(name)
        #Unfreeze batch normalization layers
        BNFrozentoBN(model, trainConvBlocks)

    elif trainmode == "RESNETF":
        trainConvBlocks = BN_LAYERS['RESNETF'] #[start,end] with end inclusive
        layerNoFreeze = []
        #Adding resnet layers which should be trained/ not freezed
        for l_name,indices in trainConvBlocks.items():
            layerNoFreeze.extend(["backbone.bottom_up.{}.{}".format(l_name,i) for i in range(indices[0],indices[1]+1)])
        print("Layers to train: ",layerNoFreeze)
        for name, param in list(model.named_parameters()):
            isTrainLayer = any([name.find(l_name) != -1 for l_name in layerNoFreeze])
            if not isTrainLayer:
                param.requires_grad = False     #freeze layer
                print("Freeze layer: ",name)
            else:
                print("Train layer: ",name)
                trainlayers.append(name)
        #Unfreeze batch normalization layers
        BNFrozentoBN(model, trainConvBlocks)

    elif trainmode == 'HEADSALL':
        layersNoFreezePrefix = [PREFIXES_LAYERNAMES[x] for x in BN_LAYERS['HEADSALL']]
        for name, param in list(model.named_parameters()):
            isTrainLayer = any([name.find(l_name) != -1 for l_name in layersNoFreezePrefix])
            if not isTrainLayer:
                print("Freeze layer: ",name)
                param.requires_grad = False     #freeze layer
            else:
                print("Train layer: ",name)
                trainlayers.append(name)

    elif trainmode == 'EVALBASELINE':
        for name, param in list(model.named_parameters()):
            if name != 'roi_heads.keypoint_head.score_lowres.weight':
                param.requires_grad = False     #freeze layer
    """elif trainmode == 'FPN+HEADS':
        layersNoFreezePrefix = [PREFIXES_LAYERNAMES[x] for x in ["FPN","ROI_KEYPOINT","ROI_PREDICTOR","ROI_HEAD","RPN_HEAD","RPN_ANCHOR"]]
        for name, param in list(model.named_parameters()):
            isTrainLayer = any([name.find(l_name) != -1 for l_name in layersNoFreezePrefix])
            if not isTrainLayer:
                print("Not training layer: ",name)
                param.requires_grad = False     #freeze layer
            else:
                print("Train layer: ",name)
                trainlayers.append(name)"""  
    return trainlayers

def get_checkparams(model, trainlayers):
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
    return checkParams

def check_modelparams(model, checkParams):
        #Check if some freezed layers have the same weights as before training
        for name, param in list(model.named_parameters()):
            if name in checkParams:
                if not torch.equal(param.data, checkParams[name].data):
                    print("Warning: Layer %s has been modified whereas it should actually been freezed."%name)

def save_modelconfigs(outdir, cfg, LAYERSBN_map, args):
    filename = 'run_configs.csv'
    filepath = os.path.join(outdir, filename)

    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            headers = ['Folder', 'NET', 'BN', 'LR', 'Gamma', 'Steps', 'Epochs', 'Data Augmentation [CropSize, FlipProb, RotationAngle]',
                       'Min Keypoints', 'MinSize Train', 'ImPerBatch', 'Additional']
            writer.writerow(headers)
    folder = os.path.basename(cfg.OUTPUT_DIR)
    bnlayers = LAYERSBN_map[args.finetune]
    data_augm = [cfg.INPUT.CROP.SIZE, cfg.DATA_FLIP_PROBABILITY , cfg.ROTATION]  #TODO: add additional items to cfg object
    lr = '%.2E'%Decimal(str(cfg.SOLVER.BASE_LR))

    row = [folder, args.finetune, bnlayers, lr, str(cfg.SOLVER.GAMMA), cfg.SOLVER.STEPS,
            args.numepochs, data_augm, cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE, cfg.INPUT.MIN_SIZE_TRAIN, cfg.SOLVER.IMS_PER_BATCH ]
    with open(filepath, 'a') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(row)
    print("Sucessfully wrote hyper-parameter row to configs file.")

if __name__=="__main__":
    main()