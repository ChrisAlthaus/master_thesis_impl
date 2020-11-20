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
_NUMGPUS = 2

_PREDICTOR = ['MotifPredictor', 'IMPPredictor', 'VCTreePredictor', 'TransformerPredictor', 'CausalAnalysisPredictor']
_FUSION_TYPES = ['sum', 'gate']
_CONTEXTLAYER_TYPES = ['motifs', 'vctree', 'vtranse']
_LR = [0.01, 0.001, 0.005] #original: 0.01

_IMS_PER_BATCH = 12
#scalefactor to keep same number of epochs
scalefactor = 12/_IMS_PER_BATCH 
_ADD_ITER = 25000
_MAX_ITER = (50000 + _ADD_ITER) * scalefactor
_STEPS = ((10000 + _ADD_ITER)* scalefactor, (16000 + _ADD_ITER)* scalefactor) #not used
_GAMMA = 0.75 #original: 0.1
_VAL_PERIOD = 5000 * scalefactor
_CPKT_PERIOD = 20000 * scalefactor
_MINSIZES_TRAIN = (640, 672, 704, 736, 768, 800) #detectron2: (640, 672, 704, 736, 768, 800), original: (600,)

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
        lr = 0.01


    gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun%d-2.sh'%_NUMGPUS
    jobname = 'scenegraph-train%s'%datetime.datetime.now().strftime('%d_%H-%M-%S')
    masterport = random.randint(10020, 10100)

    out_dir = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/sgdet_training'
    #Using pre-trained Faster-RCNN
    pretrained_frcnn = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/faster_rcnn_training/11-12_19-22-41/model_final.pth'
    glovedir = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/sgdet_training/glove'

    logdir = os.path.join('/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/sgdet_training/logs', datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
    logfile = os.path.join(logdir, 'train.log')
    os.makedirs(logdir)

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
			'testbatchsize': 2, 
			'lr': lr, 
			'steps': tuple(map(int,_STEPS)), 
            'gamma': _GAMMA,
            'minscales': _MINSIZES_TRAIN,
			'maxiterations': int(_MAX_ITER), 
			'dtype': 'float16', 
			'valperiod': int(_VAL_PERIOD), 
			'cpktperiod': int(_CPKT_PERIOD),
            'glovedir': glovedir,
            'pretrainedcpkt': pretrained_frcnn,
			'preval': True, #False, 
			'outputdir': out_dir, 
			'addnotes': _ADD_NOTES}

    paramconfig = os.path.join(logdir, 'paramsconfig.txt')
    with open(paramconfig, 'w') as f:
            json.dump(params, f)

    #sgdet: parameter MODEL.ROI_RELATION_HEAD.REQUIRE_BOX_OVERLAP = True
    #Print console cmd for better debugging
    cmd = ("sbatch -w devbox4 -J {} -o {} "+ \
    "{} python3.6 -m torch.distributed.launch --master_port {} --nproc_per_node={} "+\
	"/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/tools/relation_train_net.py \t"+\
    "--config-file \"/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/configs/e2e_relation_X_101_32_8_FPN_1x.yaml\" \t"+\
	"--paramsconfig {}")\
		.format(jobname, logfile, gpu_cmd, masterport, _NUMGPUS, paramconfig)
        
    print(cmd)
    os.system(cmd)
    time.sleep(10)