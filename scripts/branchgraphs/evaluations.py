import os
import json 
import argparse
import numpy as np
import math
import pickle
import datetime
import time
import logging
import itertools

parser = argparse.ArgumentParser()
args = parser.parse_args()

def latestdir(dir):
    diritems = [os.path.join(dir, d) for d in os.listdir(dir)]
    all_subdirs = [d for d in diritems if os.path.isdir(d)]
    return max(all_subdirs, key=os.path.getmtime)

def filewithname(dir, searchstr):
    for item in os.listdir(dir):
        if os.path.isfile(os.path.join(dir,item)) and searchstr in item:
            return os.path.join(dir,item)
    return None


# ---------------------------- PREDICTION STATISTICS ------------------------------
print("PREDICTION STATISTICS:")
predfile = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/out/predictions/graphs/09-28_12-23-05/custom_prediction.json'

cmd = "python3.6 /home/althausc/master_thesis_impl/scripts/scenegraph/utils/graph_statistics.py -graphfile {}"\
                                                .format(predfile)
print(cmd)

outrun_dir = os.path.join(os.path.dirname(predfile), '.stats')
print("Output Directory: %s\n"%outrun_dir)


# --------------------------- VISUALIZE SCENE GRAPH PREDICTIONS --------------------------
print("VISUALIZE SCENE GRAPHS:")
preddir = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/out/predictions/single/10-12_11-57-33'
filterbylabels = False
boxestopk = 20
relstopk = 20

cmd = "python3.6 /home/althausc/master_thesis_impl/scripts/scenegraph/visualizeimgs.py -predictdir {} {} -boxestopk {} -relstopk {}"\
                                                .format(preddir, '-filter' if filterbylabels else '', boxestopk, relstopk)
print(cmd)

outrun_dir = os.path.join(preddir, '.visimages')
print("Output Directory: %s\n"%outrun_dir)  


# --------------------------------- EVALUATE MODEL --------------------------------------
print("EVLUATE MODEL:")
gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun1-2.sh'
modeldir = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/sgdet_training/12-02_09-23-52-dev3'
configfile = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/configs/e2e_relation_X_101_32_8_FPN_1x.yaml'
glovedir = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/sgdet_training/glove'
valset = "VG_styletransfer_val_subset_val" #("VG_styletransfer_val",)

jobname = 'sg-eval'
logdir = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/others/evallogs'
logfile = os.path.join(logdir, '{}.log'.format(datetime.datetime.now().strftime('%m-%d_%H-%M-%S')))

cmd = ("sbatch -w devbox4 -J {} -o {} "+ \
       "{} python3.6 /home/althausc/master_thesis_impl/scripts/scenegraph/utils/evalmodel.py -config_file {} "+\
                                                            "-dataset {} MODEL.PRETRAINED_DETECTOR_CKPT {}  GLOVE_DIR {}")\
                                                .format(jobname, logfile, gpu_cmd, configfile, valset, modeldir, glovedir)
print(cmd)

outrun_dir = os.path.join(modeldir, '.eval')
print("Output Directory: %s\n"%outrun_dir) 


