import random
import numpy as np
import datetime
import json
import os
import time
import csv

_NUMGPUS = 2 #default parameters for 4 GPUs
_IMS_PER_BATCH = 2 #default: 8
#scalefactor to get right learning rate & keep same number of epochs
scalefactor = 8/_IMS_PER_BATCH 
_LR = 0.001/scalefactor#0.0025 #original: 0.001
_MAX_ITER = 50000 * scalefactor
_STEPS = (30000* scalefactor, 45000* scalefactor) 
_VAL_PERIOD = 2000 * scalefactor
_CPKT_PERIOD = 2000 * scalefactor

_DATASET_SELECTS = ['trainandval-subset', 'val-subset', 'default-styletransfer', 'default-vg']
_DATASET_SELECT = _DATASET_SELECTS[1]

gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun2-2.sh'
jobname = 'scenegraph-train%s'%datetime.datetime.now().strftime('%d_%H-%M-%S')
masterport = random.randint(10020, 10100)

out_dir = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/faster_rcnn_training'

logdir = os.path.join('/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/faster_rcnn_training/logs', datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
logfile = os.path.join(logdir, 'train.log')
os.makedirs(logdir)

cmd = ("sbatch -w devbox4 -J {} -o {} "+ \
    "{} python3.6 -m torch.distributed.launch --master_port {} --nproc_per_node={} "+\
	"/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/tools/detector_pretrain_net.py \t"+\
	"--config-file \"/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/configs/e2e_relation_detector_X_101_32_8_FPN_1x.yaml\" \t"+\
	"SOLVER.IMS_PER_BATCH {} \t"+\
	"TEST.IMS_PER_BATCH 4 \t"+\
	"DTYPE \"float16\" \t"+\
	"SOLVER.BASE_LR {} \t"+\
	"SOLVER.MAX_ITER {} \t"+\
	"SOLVER.VAL_PERIOD {} \t"+\
	"SOLVER.CHECKPOINT_PERIOD {} \t"+\
	"DATASETS.SELECT {} \t"+\
	"MODEL.RELATION_ON False  \t"+\
	"SOLVER.PRE_VAL False \t"+\
	"OUTPUT_DIR {}")\
		.format(jobname, logfile, gpu_cmd, masterport, _NUMGPUS, int(_IMS_PER_BATCH), _LR, int(_MAX_ITER), int(_VAL_PERIOD), int(_CPKT_PERIOD), _DATASET_SELECT, out_dir)
print(cmd)
os.system(cmd)
time.sleep(10)