import random
import numpy as np
import datetime
import json
import os
import time
import csv

_NUMGPUS = 2#1 #default parameters for 4 GPUs
_IMS_PER_BATCH = 4#8 #2 #default: 8
#scalefactor to get right learning rate & keep same number of epochs
scalefactor = 2#8/_IMS_PER_BATCH 
_LR = 0.002/_IMS_PER_BATCH #0.00125 #0.0005 #0.0015 #0.00075 #0.001/scalefactor#0.0025 #original: 0.001
_ADD_ITER = 25000
_MAX_ITER = (50000 + _ADD_ITER) * scalefactor
_STEPS_PERC = [0.5, 0.7, 0.8, 0.9, 0.95] #[0.4, 0.6, 0.8, 0.9, 0.95]#original: [0.6, 0.9]
_STEPS = [(perc * _MAX_ITER + _ADD_ITER)* scalefactor for perc in _STEPS_PERC]
_GAMMA = 0.5 #original: 0.1
_VAL_PERIOD = 7500 * scalefactor
_CPKT_PERIOD = 7500 * scalefactor
_MINSIZES_TRAIN = (600,) #(640, 672, 704, 736, 768, 800) #detectron2: (640, 672, 704, 736, 768, 800), original: (600,)

# Default Training Command:
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 10001 --nproc_per_node=4 
# tools/detector_pretrain_net.py --config-file "configs/e2e_relation_detector_X_101_32_8_FPN_1x.yaml" 
# SOLVER.IMS_PER_BATCH 8 
# TEST.IMS_PER_BATCH 4 
# DTYPE "float16" 
# SOLVER.MAX_ITER 50000 
# SOLVER.STEPS "(30000, 45000)" 
# SOLVER.VAL_PERIOD 2000 
# SOLVER.CHECKPOINT_PERIOD 2000 
# MODEL.RELATION_ON False 
# OUTPUT_DIR /home/kaihua/checkpoints/pretrained_faster_rcnn 
# SOLVER.PRE_VAL False

_DATASET_SELECTS = ['trainandval-subset', 'val-subset', 'default-styletransfer', 'default-vg']
_DATASET_SELECT = _DATASET_SELECTS[1]
_ADD_NOTES = 'X-101-FPN-32-8 (like cpkt) & Lr now really like previous run & More LR-steps with gamma(0.5) & GN enabled! & Like /faster_rcnn_training/11-12_19-22-41 (Default MinSize-Step because of low validation acc on multiple, ...) & Right Scale factor(not like last run before='

resume_cpkt = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/faster_rcnn_training/11-12_19-22-41/model_final.pth'
resume = False #True #False

gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun%d-2.sh'%_NUMGPUS #'/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun1-2-qrtx8000.sh' #
jobname = 'scenegraph-train%s'%datetime.datetime.now().strftime('%d_%H-%M-%S')
masterport = random.randint(10020, 10100)

out_dir = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/faster_rcnn_training'

logdir = os.path.join('/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/faster_rcnn_training/logs', datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
logfile = os.path.join(logdir, 'train.log')
os.makedirs(logdir)

params = {'datasetselect': _DATASET_SELECT, 
			'gpus': _NUMGPUS, 
			'trainbatchsize': int(_IMS_PER_BATCH), 
			'testbatchsize': 2, #4, 
			'lr': _LR, 
			'steps': tuple(map(int,_STEPS)), 
			'gamma': _GAMMA,
			'minscales': _MINSIZES_TRAIN,
			'maxiterations': int(_MAX_ITER), 
			'dtype': 'float16', 
			'valperiod': int(_VAL_PERIOD), 
			'cpktperiod': int(_CPKT_PERIOD), 
			'relationon': False, 
			'preval': False, 
			'outputdir': out_dir,
			'resumecpkt': resume_cpkt if resume else '', 
			'addnotes': _ADD_NOTES}

paramconfig = os.path.join(logdir, 'paramsconfig.txt')
with open(paramconfig, 'w') as f:
        json.dump(params, f)

#Note: X-101-32-8-FPN ~2GB more RAM than R-101-FPN architecture
#	  -> X-101-32-8-FPN and batchsize of 4 ~ 10000MiB
configfile = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/configs/e2e_relation_detector_X_101_32_8_FPN_1x.yaml' 
				#'/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/configs/e2e_relation_detector_R_101_FPN_1x.yaml'

cmd = ("sbatch -w devbox4 -J {} -o {} "+ \
    "{} python3.6 -m torch.distributed.launch --master_port {} --nproc_per_node={} "+\
	"/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/tools/detector_pretrain_net.py \t"+\
	"--config-file {} \t"+\
	"--paramsconfig {}")\
		.format(jobname, logfile, gpu_cmd, masterport, _NUMGPUS, configfile, paramconfig)
print(cmd)
os.system(cmd)
time.sleep(10)