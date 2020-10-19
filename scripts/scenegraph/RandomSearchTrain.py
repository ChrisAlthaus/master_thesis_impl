import random
import numpy as np
import datetime
import json
import os
import time
import csv

#Perform random search over the hyper-parameters
_PARAM_MODES = ['randomsearch', 'custom']
_PARAM_MODE = _PARAM_MODES[0]
_NUM_RUNS = 2

_PREDICTOR = ['MotifPredictor', 'IMPPredictor', 'VCTreePredictor', 'TransformerPredictor', 'CausalAnalysisPredictor']
_FUSION_TYPES = ['sum', 'gate']
_CONTEXTLAYER_TYPES = ['motifs', 'vctree', 'vtranse']
_LR = [0.01, 0.001, 0.005] #original: 0.01
_IMS_PER_BATCH = 12
_MAX_ITER = 50000
_VAL_PERIOD = 2000
_CPKT_PERIOD = 2000


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

for i in range(0, _NUM_RUNS):   
    if _PARAM_MODE == 'randomsearch':
        while True:
            predictor = _PREDICTOR[4]
            fusiontype = random.choice(_FUSION_TYPES)
            contextlayer = random.choice(_CONTEXTLAYER_TYPES)
            lr = random.choice(_LR)

            if not paramsexist(predictor, fusiontype, contextlayer, lr):
                break
            print("----------------------------------")
        exit(1)
            
    elif _PARAM_MODE == 'custom':
        predictor = _PREDICTOR[4]
        fusiontype = 'sum'
        contextlayer = 'motifs'
        lr = 0.01

    gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun_G2d4-2.sh'

    out_dir = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints'
    #Using pre-trained Faster-RCNN
    pretrained_frcnn = 'master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/pretrained_faster_rcnn/model_final.pth'
    glovedir = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/glove'

    #sgdet: parameter MODEL.ROI_RELATION_HEAD.REQUIRE_BOX_OVERLAP = True
    #Print console cmd for better debugging
    cmd = ("{} python3.6 -m torch.distributed.launch --master_port 10026 --nproc_per_node=2 "+\
    	"/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/tools/relation_train_net.py \t"+\
    	"--config-file \"/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/configs/e2e_relation_X_101_32_8_FPN_1x.yaml\" \t"+\
    	"MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \t"+\
    	"MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \t"+\
    	"MODEL.ROI_RELATION_HEAD.PREDICTOR {} \t"+\
    	"MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE none \t"+\
    	"MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE {} \t"+\
    	"MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER {} \t"+\
    	"MODEL.ROI_RELATION_HEAD.REQUIRE_BOX_OVERLAP True \t"+\
    	"SOLVER.IMS_PER_BATCH {} \t"+\
    	"TEST.IMS_PER_BATCH 2 \t"+\
    	"DTYPE \"float16\" \t"+\
    	"SOLVER.MAX_ITER {} \t"+\
    	"SOLVER.VAL_PERIOD {} \t"+\
    	"SOLVER.CHECKPOINT_PERIOD {} \t"+\
    	"GLOVE_DIR {} \t"+\
    	"MODEL.PRETRAINED_DETECTOR_CKPT {} \t"+\
    	"OUTPUT_DIR {}")\
    		.format(gpu_cmd, predictor, fusiontype, contextlayer, _IMS_PER_BATCH, _MAX_ITER, _VAL_PERIOD, _CPKT_PERIOD, glovedir, pretrained_frcnn, out_dir)
    print(cmd)

    gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun2-2.sh'
    jobname = 'scenegraph-train%d'%i
    logdir = os.path.join('/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/logs', datetime.datetime.now().strftime('%m-%d_%H-%M-%S')+ '_%d'%i)
    os.makedirs(logdir)

    cmd = ("sbatch -w devbox4 -J {} -o {} "+ \
        "{} python3.6 -m torch.distributed.launch --master_port 10026 --nproc_per_node=2 "+\
    	"/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/tools/relation_train_net.py \t"+\
    	"--config-file \"/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/configs/e2e_relation_X_101_32_8_FPN_1x.yaml\" \t"+\
    	"MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \t"+\
    	"MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \t"+\
    	"MODEL.ROI_RELATION_HEAD.PREDICTOR {} \t"+\
    	"MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE none \t"+\
    	"MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE {} \t"+\
    	"MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER {} \t"+\
    	"MODEL.ROI_RELATION_HEAD.REQUIRE_BOX_OVERLAP True \t"+\
    	"SOLVER.IMS_PER_BATCH {} \t"+\
    	"TEST.IMS_PER_BATCH 2 \t"+\
    	"DTYPE \"float16\" \t"+\
    	"SOLVER.MAX_ITER {} \t"+\
    	"SOLVER.VAL_PERIOD {} \t"+\
    	"SOLVER.CHECKPOINT_PERIOD {} \t"+\
    	"GLOVE_DIR {} \t"+\
    	"MODEL.PRETRAINED_DETECTOR_CKPT {} \t"+\
    	"OUTPUT_DIR {}")\
    		.format(jobname, logdir, gpu_cmd, predictor, fusiontype, contextlayer, _IMS_PER_BATCH, _MAX_ITER, _VAL_PERIOD, _CPKT_PERIOD, glovedir, pretrained_frcnn, out_dir)
    print(cmd)
    #os.system(cmd)
    time.sleep(10)