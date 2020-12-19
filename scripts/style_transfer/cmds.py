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


# ---------------------------- STYLE TRANSFER ------------------------------
print("APPLY STYLE TRANSFER:")
styledir = '/nfs/data/iart/kaggle/img'  #kaggle painters by numbers dataset
imagesdir = '/nfs/data/coco_17/train2017'
outputdir = '/home/althausc/nfs/data/coco_17_large/train2017_styletransfer' #val2017
numcontents = -1
numstyles = 2
splitfraction = 1.0 #0.85
splitdirection = 'begin'
verbose = False #only for debugging
imgoutnaming = 'aggregate' #aggregate-> content_style.jpg , keepcontent-> content.jpg (numstyles=1)

gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/tensorflow_srun1.sh'
jobname = 'styletransfer'
logfile = os.path.join(os.path.dirname(outputdir), 'log-%s.txt'%'train')

cmd = ("sbatch -w devbox4 -J {} -o {} "+ \
        "{} python3.6 /home/althausc/master_thesis_impl/scripts/style_transfer/fast_style_transfer.py -styles {} -images {} -o {} "+ \
                                                                     "-nC {} -nS {} -split {} -splitdirection {} -labelmode {} {}")\
                        .format(jobname, logfile, gpu_cmd, styledir, imagesdir, outputdir, numcontents, numstyles, splitfraction, splitdirection, imgoutnaming, '-v' if verbose else '')
print(cmd)

print("Output Directory: %s\n"%outputdir)


# --------------------------- GET STYLETRANSFER ANNOTATIONS --------------------------
print("VISUALIZE SCENE GRAPHS:")
annotationfile = '/nfs/data/coco_17/annotations/person_keypoints_train2017.json'  #person_keypoints_val2017.json
imagedir_styletransfered = outputdir
outputdir = '/home/althausc/nfs/data/coco_17_large/annotations_styletransfer'
annmode = 'COCOAPI'

cmd = ("python3.6 /home/althausc/master_thesis_impl/scripts/style_transfer/get_updated_coco_annotations.py"+ \
                "-jsonAnnotation {} -styleTransferImDir {} -outputDirectory {} --annotationSchema {}")\
                                                .format(annotationfile, imagedir_styletransfered, outputdir, annmode)
print(cmd)

print("Output Directory: %s\n"%outputdir)  
