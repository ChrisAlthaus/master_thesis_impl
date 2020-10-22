import random
import numpy as np
import datetime
import json
import os
import time
import csv

#Perform random search over the hyper-parameters
['Folder', 'LR', 'BN', 'BatchSize', 'PretrainedModel', 'Epochs', 'LrDecayEpochs', 'LrDecayFactor', 'Loss_h', 'Loss_c', 'Add.Notes']

_LR = [0.0005, 0.001, 0.00025] #original: 5e-4
_BN = True
_BATCHSIZES = [16, 8] #original: 16
_EPOCHNUM = 140 #original: 140
_LRDECAY_FACTORS = 10 #original: 10
_LRDECAY_EPOCHS = [[90, 120]] #original: [90, 120]


#Grid Search not implemented so far, set parameters manually in PoseFix_RELEASE/main/config.py
#Normal run command, Print for eventually debugging
gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/tensorflow_srun-G1D4.sh'
resume = False #True
pretrained_dir = 'test'
inputpreds = '/home/althausc/master_thesis_impl/detectron2/out/art_predictions/train/10-22_13-25-10/maskrcnn_predictions.json'

cmd = ("{} python3.6 /home/althausc/master_thesis_impl/PoseFix_RELEASE/main/train.py --gpu 1 --inputpreds {} {}")\
                                                .format(gpu_cmd, inputpreds, '--continue --pretrained %s'%pretrained_dir if resume else ' ')
print(cmd)
print()

#Start sbatch training
gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/tensorflow_srun1.sh '
jobname = "posefix-train-%s"%datetime.datetime.now().strftime('%d_%H-%M-%S')

logdir = os.path.join('/home/althausc/master_thesis_impl/PoseFix_RELEASE/output/log/COCO/logs', datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
os.makedirs(logdir)
logfile = os.path.join(logdir, 'train.log')

cmd = ("sbatch -w devbox4 -J {} -o {} "+ \
    "{} python3.6 /home/althausc/master_thesis_impl/PoseFix_RELEASE/main/train.py --gpu 1 --inputpreds {} {}")\
                                                .format(jobname, logfile, gpu_cmd, inputpreds, '--continue --pretrained %s'%pretrained_dir if resume else ' ')
print(cmd)
os.system(cmd)
time.sleep(10)