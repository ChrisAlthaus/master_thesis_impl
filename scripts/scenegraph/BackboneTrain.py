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
_LR = 0.001 #0.0015 #0.00075 #0.001/scalefactor#0.0025 #original: 0.001
_ADD_ITER = 25000
_MAX_ITER = (50000 + _ADD_ITER) * scalefactor
_STEPS = ((30000 + _ADD_ITER)* scalefactor, (45000 + _ADD_ITER)* scalefactor) 
_VAL_PERIOD = 4000 * scalefactor
_CPKT_PERIOD = 4000 * scalefactor

_DATASET_SELECTS = ['trainandval-subset', 'val-subset', 'default-styletransfer', 'default-vg']
_DATASET_SELECT = _DATASET_SELECTS[3]
_ADD_NOTES = ''

gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun%d-2.sh'%_NUMGPUS
jobname = 'scenegraph-train%s'%datetime.datetime.now().strftime('%d_%H-%M-%S')
masterport = random.randint(10020, 10100)

out_dir = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/faster_rcnn_training'

logdir = os.path.join('/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/faster_rcnn_training/logs', datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
logfile = os.path.join(logdir, 'train.log')
os.makedirs(logdir)

params = {'datasetselect': _DATASET_SELECT, 
			'gpus': _NUMGPUS, 
			'trainbatchsize': int(_IMS_PER_BATCH), 
			'testbatchsize': 4, 
			'lr': _LR, 
			'steps': tuple(map(int,_STEPS)), 
			'maxiterations': int(_MAX_ITER), 
			'dtype': 'float16', 
			'valperiod': int(_VAL_PERIOD), 
			'cpktperiod': int(_CPKT_PERIOD), 
			'relationon': False, 
			'preval': False, 
			'outputdir': out_dir, 
			'addnotes': _ADD_NOTES}

paramconfig = os.path.join(logdir, 'paramsconfig.txt')
with open(paramconfig, 'w') as f:
        json.dump(params, f)


cmd = ("sbatch -w devbox4 -J {} -o {} "+ \
    "{} python3.6 -m torch.distributed.launch --master_port {} --nproc_per_node={} "+\
	"/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/tools/detector_pretrain_net.py \t"+\
	"--config-file \"/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/configs/e2e_relation_detector_X_101_32_8_FPN_1x.yaml\" \t"+\
	"--paramsconfig {}")\
		.format(jobname, logfile, gpu_cmd, masterport, _NUMGPUS, paramconfig)
print(cmd)
os.system(cmd)
time.sleep(10)