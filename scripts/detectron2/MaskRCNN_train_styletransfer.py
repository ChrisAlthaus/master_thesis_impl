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

from tempfile import NamedTemporaryFile
import shutil
import json

from plotAveragePrecisions import plotAPS

from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import torch
from decimal import Decimal

import argparse

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('-finetune','-fL',required=True, 
    #                    help='Specify which layers should be trained. Either RESNET, HEADSALL or ALL.')
    parser.add_argument('-resume','-checkpoint', 
                        help='Train model from checkpoint given by path.')
    #parser.add_argument('-numepochs','-epochs', type=int, help='Number of epochs to train.')
    parser.add_argument('-paramsconfig', type=str, help='Path to config which contains specific hyper-parameters, with which the model should be trained.')
    parser.add_argument('-addconfig', help='Add selected configurations as an additional row to a csv file.', action="store_true")

    
    args = parser.parse_args()
    
    print("Reading config (hyper-)parameters from file: ",args.paramsconfig)
    c_params = []
    with open (args.paramsconfig, "r") as f:
        c_params = json.load(f)
    
    trainmode = c_params['trainmode']
    if trainmode not in ["RESNETF", "RESNETL", "HEADSALL", "ALL",'EVALBASELINE','FPN+HEADS','SCRATCH']:
        raise ValueError("Not specified a valid training mode for layers.")
    if args.resume is not None:
        if not os.path.isfile(args.resume):
            raise ValueError("Checkpoint does not exists.")

    
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")) #"COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"
    
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    if trainmode != 'SCRATCH':
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml") #"COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"
    else:
        cfg.MODEL.WEIGHTS = ""

    cfg.MODEL.DEVICE='cuda' #'cpu'   


    output_dir = os.path.join('/home/althausc/master_thesis_impl/detectron2/out/checkpoints', datetime.datetime.now().strftime('%m-%d_%H-%M-%S_'+trainmode.lower()))#+))
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

    """train_ann = "/nfs/data/coco_17/annotations/person_keypoints_train2017.json"
    train_dir = "/nfs/data/coco_17/train2017"
    val_ann = "/nfs/data/coco_17/annotations/person_keypoints_val2017.json"
    val_dir = "/nfs/data/coco_17/val2017"""

    
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
    cfg.INPUT.CROP.ENABLED = c_params['dataaugm'] #False #True
    
    cfg.DATA_FLIP_PROBABILITY = 0.25 
    cfg.DATA_FLIP_ENABLED = c_params['dataaugm'] #False #True
    cfg.ROTATION = [-15,15]
    cfg.ROTATION_ENABLED = c_params['dataaugm'] #False #True
    cfg.COLOR_AUGM_ENABLED = c_params['dataaugm'] #False #True
    
    cfg.INPUT.MIN_SIZE_TRAIN = tuple(c_params['minscales'])  #512  #Defaults: (640, 672, 704, 736, 768, 800) #Size of the smallest side of the image during training
        	                    
    
    # ------------------------ SPECIFIC LAYER PARAMETERS -------------------------
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 # 2 faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon) ?
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 17
    cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE = c_params['minkpt'] #1 #10 # Images with too few (or no) keypoints are excluded from training (default: 1)
    #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model, only for test?!
    cfg.MODEL.RPN.POSITIVE_FRACTION = c_params['rpn_posratio'] #default: 0.5


    # ---------------------------- SOLVER PARAMETERS ---------------------------
    
    cfg.SOLVER.IMS_PER_BATCH = c_params['batchsize'] #2 #4, original=2 # Number of images per batch across all machines.
    cfg.SOLVER.BASE_LR = c_params['lr'] #0.02/8#0.0025, original=0.001  # pick a good LR   #TODO: different values
    cfg.SOLVER.GAMMA = c_params['gamma']
    
    
    max_iter, epoch_iter = get_iterations_for_epochs(dataset, c_params['epochs'], cfg.SOLVER.IMS_PER_BATCH, cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE)
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.TEST.EVAL_PERIOD = int(epoch_iter/2)   #Evaluation once at the end of each epoch, Set to 0 to disable.
    cfg.TEST.PLOT_PERIOD = int(epoch_iter) # Plot val & train loss curves at every second iteration 
                                                # and save as image in checkpoint folder. Disable: -1
    cfg.SOLVER.EARLYSTOPPING_PERIOD = int(epoch_iter * 1) #window size
    cfg.TEST.PERIODICWRITER_PERIOD = 100# default:20
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = c_params['gradient_clipvalue']
    
    #steps_exp = np.linspace(0,1,12)[1:-1] * max_iter
    #For lr decay from a specific iteration number
    #iternum = 191893
    #steps_exp = np.linspace(iternum/max_iter,1,12)[0:-1] * max_iter
    #(int(5/10*max_iter),int(6/10*max_iter), int(7/10*max_iter),int(8/10*max_iter),int(9/10*max_iter),int(98/100*max_iter))
    #(int(7/9*max_iter), int(8/9*max_iter))#(int(55/90*max_iter),int(75/90*max_iter),int(85/90*max_iter))#(int(7/9*max_iter), int(8/9*max_iter)) # The iteration marks to decrease learning rate by GAMMA.  
    #steps_exp.astype(np.int)
    cfg.SOLVER.STEPS = tuple([x * max_iter for x in c_params['steps']])
    

    # -------------------------- BATCH NORMALIZATION SETUP ------------------------
    setupLayersAndBN(cfg, trainmode, batchnorm= c_params['bn'])
    cfg.freeze()
    

    # ------------------------- APPLY CONFIG & GET MODEL ------------------------
    trainer = COCOTrainer(cfg)    
    model = trainer.model
    

    # ----------------------------- SAVING CONFIGS -------------------------------
    if args.addconfig:
        print("Add reduced config to overview file: ")
        save_modelconfigs(cfg, c_params) 
        
    print("Save additional hyper-parameter: ")
    with open(os.path.join(cfg.OUTPUT_DIR, 'configparams.txt'), 'w') as f:
        json.dump(c_params, f)

    print("Save model's state_dict:")
    with open(os.path.join(cfg.OUTPUT_DIR, 'layer_params_overview.txt'), 'w') as f:
        for name, param in list(model.named_parameters()):
            f.write('{} requires_gradient: {}'.format(name, param.requires_grad)+ os.linesep)

    print("Save model's architecture:")
    with open(os.path.join(cfg.OUTPUT_DIR, 'model_architectur.txt'), 'w') as f:
        print(list(model.children()),file=f)

    print("Save model's configuration:")
    with open(os.path.join(cfg.OUTPUT_DIR, 'model_conf.txt'), 'w') as f:
        print(cfg,file=f)
        

    # -------------------- RESUME MODEL ON/OFF ---------------------
  
    if args.resume is not None:
        print("Resuming training from checkpoint %s."%args.resume)
        DetectionCheckpointer(trainer.model).load(args.resume)
    
    #trainer.resume_or_load(resume=False)

    # --------------------- TRAIN MODEL ----------------------------------
    checkparams = get_checkparams(model)

    print("START TRAINING")
    trainer.train()
    print("TRAINING DONE.")

    check_modelparams(model, checkparams)
    #Add losses to config overview file
    addlosses_to_configs(cfg.OUTPUT_DIR)

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


def setupLayersAndBN(cfg, trainmode, batchnorm=False):
    #Freeze specific layers which should not be trained according to trainmode
    #Additional set BN of ResNet

    #add custom entry to config
    cfg.MODEL.BACKBONE.FREEZE_FROM = 0
    cfg.MODEL.FPN.FREEZE = False

    #Freeze layers according to trainmode      
    if trainmode == "ALL" or trainmode == 'SCRATCH':
        cfg.MODEL.BACKBONE.FREEZE_AT = 0 # `1` means freezing the stem. `2` means freezing the stem and one residual stage, etc.
        cfg.MODEL.BACKBONE.FREEZE_AT_ENABLED = True
        cfg.MODEL.BACKBONE.FREEZE_FROM_ENABLED = False
        
        
    elif trainmode == "RESNETL":
        cfg.MODEL.BACKBONE.FREEZE_AT = 3 #Freezing first 3 resnet layers (including stem)
        cfg.MODEL.BACKBONE.FREEZE_AT_ENABLED = True
        cfg.MODEL.BACKBONE.FREEZE_FROM_ENABLED = False
        cfg.MODEL.FPN.FREEZE = True

    elif trainmode == "RESNETF":
       cfg.MODEL.BACKBONE.FREEZE_FROM = 4 #Train first 3 resnet layers (including stem)
       cfg.MODEL.BACKBONE.FREEZE_AT_ENABLED = False
       cfg.MODEL.BACKBONE.FREEZE_FROM_ENABLED = True

    elif trainmode == 'HEADSALL':
        cfg.MODEL.BACKBONE.FREEZE_AT = 5 #Freezing entire backbone
        cfg.MODEL.BACKBONE.FREEZE_AT_ENABLED = True
        cfg.MODEL.BACKBONE.FREEZE_FROM_ENABLED = False
        cfg.MODEL.FPN.FREEZE = True
    
    #Unfreeze batch normalization layers of ResNet
    if batchnorm:
        cfg.MODEL.RESNETS.NORM = "BN"


def get_checkparams(model):
    checkParams = dict()    #Dict to save intial parameters for some random freezed layers
                            #for later checking if really not trained            
    numLayersForCheck = 20
   
    freezedlayers = [ [name,param] for name,param in model.named_parameters() if param.requires_grad == False ]
    if len(freezedlayers) > numLayersForCheck: 
        numAdded = 0
        while(numAdded < numLayersForCheck):
            name, param = random.choice(list(model.named_parameters()))
            if param.requires_grad == True or name in checkParams.keys():
                continue
            else:
                checkParams.update({name:param})
                numAdded = numAdded + 1
    else:
        checkParams = {name:param for name,param in freezedlayers}
        
    return checkParams

def check_modelparams(model, checkParams):
        #Check if some freezed layers have the same weights as before training
        for name, param in list(model.named_parameters()):
            if name in checkParams:
                if not torch.equal(param.data, checkParams[name].data):
                    print("Warning: Layer %s has been modified whereas it should actually been freezed."%name)

def save_modelconfigs(cfg, params):
    filepath = '/home/althausc/master_thesis_impl/detectron2/out/checkpoints/run_configs.csv'

    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            headers = ['Folder', 'NET', 'BN', 'LR', 'Gamma', 'Steps', 'Epochs', 'Data Augmentation [CropSize, FlipProb, RotationAngle]',
                       'Min Keypoints', 'MinSize Train', 'ImPerBatch', 'Train Loss', 'Val Loss', 'bbox/AP', 'bbox/AP50', 'bbox/AP75', 'keypoints/AP', 'keypoints/AP50', 'keypoints/AP75', 'Add.Notes']
            writer.writerow(headers)
    folder = os.path.basename(cfg.OUTPUT_DIR)
    bnlayers = 'True' if params['bn'] else 'False'
    data_augm = 'True' if params['dataaugm'] else 'False'  #[cfg.INPUT.CROP.SIZE, cfg.DATA_FLIP_PROBABILITY , cfg.ROTATION]
    lr = '%.2E'%Decimal(str(cfg.SOLVER.BASE_LR))
    runnotes = params['addnotes']

    row = [folder, params['trainmode'], bnlayers, lr, str(cfg.SOLVER.GAMMA), cfg.SOLVER.STEPS,
            params['epochs'], data_augm, cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE, cfg.INPUT.MIN_SIZE_TRAIN, cfg.SOLVER.IMS_PER_BATCH, 
            ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', runnotes]
    with open(filepath, 'a') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(row)
    print("Sucessfully wrote hyper-parameter row to configs file.")

def addlosses_to_configs(modeldir):
    #Add Train & Val Loss of last N entries in metrics.json to current overview config entry
    #Seperate from save_modelconfigs, because maybe exception while training in a later epoch

    #Get Losses
    lines = []
    with open(os.path.join(modeldir, 'metrics.json'), 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    _LASTN = 100
    trainloss_lastn = [entry['total_loss'] for entry in lines[-_LASTN:]]
    valloss_lastn = [entry['validation_loss'] for entry in lines[-_LASTN:] if 'validation_loss' in entry]
    trainloss = np.mean(trainloss_lastn)
    valloss = np.mean(valloss_lastn)
    print("Averaged last N losses:")
    print("\tTrain Loss: ",trainloss)
    print("\tValidation Loss: ",valloss)

    #Get BBOX APS
    bbox_ap = lines[-1]["bbox/AP"] if "bbox/AP" in lines[-1] else 'not found'
    bbox_ap50 = lines[-1]["bbox/AP50"] if "bbox/AP50" in lines[-1] else 'not found'
    bbox_ap75 = lines[-1]["bbox/AP75"] if "bbox/AP75" in lines[-1] else 'not found'

    kpts_ap = lines[-1]["keypoints/AP"] if "keypoints/AP" in lines[-1] else 'not found'
    kpts_ap50 = lines[-1]["keypoints/AP50"] if "keypoints/AP50" in lines[-1] else 'not found'
    kpts_ap75 = lines[-1]["keypoints/AP75"] if "keypoints/AP75" in lines[-1] else 'not found'
    
    #Update CSV config
    csvfile = '/home/althausc/master_thesis_impl/detectron2/out/checkpoints/run_configs.csv'
    tempfile = NamedTemporaryFile('w+t', newline='', delete=False, dir='/home/althausc/master_thesis_impl/detectron2/out/checkpoints/tmp')
    shutil.copyfile(csvfile, tempfile.name)
    foldername = os.path.basename(modeldir)

    content = None
    with open(csvfile, 'r', newline='') as csvFile:
        reader = csv.reader(csvFile, delimiter='\t')
        content = list(reader)
        for i,row in enumerate(content):
            if i==0:
                header = row
                header = [h.strip() for h in header]
            else:
                #update losses in row of current model entry
                print(row[header.index('Folder')], foldername, row[header.index('Folder')] == foldername)
                if row[header.index('Folder')] == foldername:
                    row[header.index('Train Loss')] = trainloss
                    row[header.index('Val Loss')] = valloss
                    row[header.index('bbox/AP')] = bbox_ap
                    row[header.index('bbox/AP50')] = bbox_ap50
                    row[header.index('bbox/AP75')] = bbox_ap75
                    row[header.index('keypoints/AP')] = kpts_ap
                    row[header.index('keypoints/AP50')] = kpts_ap50
                    row[header.index('keypoints/AP75')] = kpts_ap75

                    break
        
        
    with open(csvfile, 'w', newline='') as csvFile:  
        writer = csv.writer(csvFile, delimiter='\t')
        writer.writerows(content)
    
    print("Backup of unchanged configs.csv at: ",tempfile.name)

if __name__=="__main__":
    main()