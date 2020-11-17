import random
import numpy as np
import datetime
import json
import os
import time
import csv

#Perform random search over the hyper-parameters
#['Folder', 'LR', 'BN', 'BatchSize', 'PretrainedModel', 'Epochs', 'LrDecayEpochs', 'LrDecayFactor', 'Loss_h', 'Loss_c', 'Add.Notes']
_NUMGPUS = 1

_LR = [0.0005, 0.001, 0.00025] 
_BN = True
_BATCHSIZES = [16, 8] 
_EPOCHNUM = 140 
_LRDECAY_FACTORS = 10 
_LRDECAY_EPOCHS = [[90, 120]] 

batchsize = 16  #original: 16
lr_dec_epoch = [90, 120]    #original: [90, 120]
epochs = 140    #original: 140
lr = 0.005  #original: 5e-4
lr_dec_factor = 10  #original: 10

_ADD_NOTES = ''

resume = False #True
resumedir = ''
inputpreds = '/home/althausc/master_thesis_impl/PoseFix_RELEASE/inputs/10-23_17-34-31/predictions_bbox_gt.json'

logdir = os.path.join('/home/althausc/master_thesis_impl/PoseFix_RELEASE/output/log/COCO/logs', datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
os.makedirs(logdir)
logfile = os.path.join(logdir, 'train.log')

params = { 'trainbatchsize': batchsize, 
			'lr': lr, 
			'steps_epochs': lr_dec_epoch, 
			'epochs': epochs, 
			'lr_dec_factor': lr_dec_factor, 
            'predictionsfile': inputpreds,
			'resumecpkt': resumedir if resume else '',
			'addnotes': _ADD_NOTES}

paramconfig = os.path.join(logdir, 'paramsconfig.txt')
with open(paramconfig, 'w') as f:
        json.dump(params, f)

#Grid Search not implemented so far, set parameters manually in PoseFix_RELEASE/main/config.py
#Normal run command, Print for eventually debugging
gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/tensorflow_srun-G%dD4.sh'%_NUMGPUS
cmd = ("{} python3.6 /home/althausc/master_thesis_impl/PoseFix_RELEASE/main/train.py --paramsconfig {}")\
                                                .format(gpu_cmd, paramconfig)
print(cmd)

#Start sbatch training
gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/tensorflow_srun%d.sh'%_NUMGPUS
jobname = "posefix-train-%s"%datetime.datetime.now().strftime('%d_%H-%M-%S')


cmd = ("sbatch -w devbox4 -J {} -o {} "+ \
    "{} python3.6 /home/althausc/master_thesis_impl/PoseFix_RELEASE/main/train.py --paramsconfig {}")\
                                               .format(jobname, logfile, gpu_cmd, paramconfig)
print(cmd)
os.system(cmd)
time.sleep(10)