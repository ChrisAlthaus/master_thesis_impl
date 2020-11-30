import random
import numpy as np
import datetime
import json
import os
import time
import csv

#Perform random search over the hyper-parameters
_PARAM_MODES = ['randomsearch', 'custom']
_PARAM_MODE = _PARAM_MODES[1]
_NUMRUNS = 1
_NUMGPUS = 4

_PREDICTOR = ['MotifPredictor', 'IMPPredictor', 'VCTreePredictor', 'TransformerPredictor', 'CausalAnalysisPredictor']
_FUSION_TYPES = ['sum', 'gate']
_CONTEXTLAYER_TYPES = ['motifs', 'vctree', 'vtranse']
_LR = [0.01, 0.001, 0.005] #original: 0.01

_IMS_PER_BATCH = 4#8
#scalefactor to keep same number of epochs -> ~15 Epochs
scalefactor = 12/_IMS_PER_BATCH 
_ADD_ITER = 25000
_MAX_ITER = (50000 + _ADD_ITER) * scalefactor   
_VAL_PERIOD = 5000 * scalefactor
_CPKT_PERIOD = 10000 * scalefactor
_MINSIZES_TRAIN = (640, 672, 704, 736, 768, 800) #detectron2: (640, 672, 704, 736, 768, 800), original: (600,)

_LR_SCHEDULER = "WarmupReduceLROnPlateau" #"WarmupMultiStepLR"
_GAMMA = ''
_STEPS = ''
_PATIENCE = ''
if _LR_SCHEDULER == "WarmupReduceLROnPlateau":
    _GAMMA = 0.75
    #Description: If recall not improves within X validation points reduce lr
    _PATIENCE = 2 
elif _LR_SCHEDULER == "WarmupMultiStepLR":
    _STEPS_PERC = [0.6, 0.8, 0.9, 0.95]
    _STEPS = [(perc * _MAX_ITER) for perc in _STEPS_PERC]
    _GAMMA = 0.75


_DATASET_SELECTS = ['trainandval-subset', 'val-subset', 'default-styletransfer', 'default-vg']
_DATASET_SELECT = _DATASET_SELECTS[2]

_ATTRIBUTE_ON = False
_ADD_NOTES = 'Additional multiple minscales like detectron2, higher gamma & data augmentation (color)'

def paramsexist(predictor, fusiontype, contextlayer, lr):
    #Look in the overview csv file containing all done training runs if random params exists
    csvfile = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/run_configs.csv'
    with open(csvfile, 'r', newline='') as csvFile:
        reader = csv.reader(csvFile, delimiter='\t')
        content = list(reader)
        header = []
        for i,row in enumerate(content):
            if i==0:
                header = row   
            else:
                if row[header.index('Predictor')] == predictor and row[header.index('Fusion')] == fusiontype and \
                    row[header.index('ContextLayer')] == contextlayer and float(row[header.index('LR')]) == lr:
                    return True
    return False

for i in range(0, _NUMRUNS):   
    if _PARAM_MODE == 'randomsearch':
        while True:
            predictor = _PREDICTOR[4]
            #Fusiontype and contextlayer only used for 'CausalAnalysisPredictor'
            fusiontype = random.choice(_FUSION_TYPES)
            contextlayer = random.choice(_CONTEXTLAYER_TYPES)
            lr = random.choice(_LR)

            if not paramsexist(predictor, fusiontype, contextlayer, lr):
                break
            print("----------------------------------")
            
    elif _PARAM_MODE == 'custom':
        predictor = _PREDICTOR[4]
        fusiontype = 'sum'
        contextlayer = 'motifs'
        lr = 0.01 / (_NUMGPUS/2)


    gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun%d-2.sh'%_NUMGPUS
    jobname = 'sg-train%s'%datetime.datetime.now().strftime('%d_%H-%M-%S')
    masterport = random.randint(10020, 10100)

    out_dir = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/sgdet_training'
    #Using pre-trained Faster-RCNN
    pretrained_frcnn = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/faster_rcnn_training/11-20_15-29-23/model_final.pth'
            #'/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/faster_rcnn_training/11-12_19-22-41/model_final.pth'
    glovedir = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/sgdet_training/glove'
    configfile = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/configs/e2e_relation_R_101_FPN_1x.yaml' 
                #'/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/configs/e2e_relation_X_101_32_8_FPN_1x.yaml' 

    logdir = os.path.join('/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/sgdet_training/logs', datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
    logfile = os.path.join(logdir, 'train.log')
    os.makedirs(logdir)

    #SGDet important settings:
    # MODEL.ROI_RELATION_HEAD.USE_GT_BOX False 
    # MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False
    # MODEL.ROI_RELATION_HEAD.REQUIRE_BOX_OVERLAP = True (optional?)
    # MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE only for inference

    params = {'datasetselect': _DATASET_SELECT, 
			'gpus': _NUMGPUS, 
            'use_gt_box': False,
            'use_gt_objectlabel': False,
            'predictor': predictor,
            'effecttype': 'none',
            'fusiontype': fusiontype,
            'contextlayer': contextlayer,
            'require_bboxoverlap': True,
            'attribute_on': _ATTRIBUTE_ON,
			'trainbatchsize': int(_IMS_PER_BATCH), 
			'testbatchsize': 2#1* _NUMGPUS, 
            'lrscheduler': _LR_SCHEDULER,
			'lr': lr, 
			'steps': tuple(map(int,_STEPS)), 
            'gamma': _GAMMA,
            'patience': _PATIENCE,
            'minscales': _MINSIZES_TRAIN,
			'maxiterations': int(_MAX_ITER), 
			'dtype': 'float16', 
			'valperiod': int(_VAL_PERIOD), 
			'cpktperiod': int(_CPKT_PERIOD),
            'glovedir': glovedir,
            'pretrainedcpkt': pretrained_frcnn,
			'preval': False, 
			'outputdir': out_dir, 
			'addnotes': _ADD_NOTES}

    paramconfig = os.path.join(logdir, 'paramsconfig.txt')
    with open(paramconfig, 'w') as f:
            json.dump(params, f, indent=4)

   
    #Print console cmd for better debugging
    cmd = ("sbatch -w devbox4 -J {} -o {} "+ \
    "{} python3.6 -m torch.distributed.launch --master_port {} --nproc_per_node={} "+\
	"/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/tools/relation_train_net.py \t"+\
    "--config-file {} \t"+\
	"--paramsconfig {}")\
		.format(jobname, logfile, gpu_cmd, masterport, _NUMGPUS, configfile, paramconfig)
        
    print(cmd)
    os.system(cmd)
    time.sleep(10)