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


# ----------------- EVALUATE MASK-RCNN PREDICTIONS WITH DETAILED EVALUATION ---------------------
print("DETAILED MASK-RCNN EVALUATION:")
predfile = '/home/althausc/master_thesis_impl/detectron2/out/art_predictions/train/10-22_13-25-10/maskrcnn_predictions.json'
gtannfile = '/home/althausc/nfs/data/coco_17_medium/annotations_styletransfer/person_keypoints_train2017_stAPI.json'
outdir = '/home/althausc/master_thesis_impl/results/posedetection/maskrcnn'
teamname = 'artinfer'
version = 1.0

cmd = "python3.6 /home/althausc/master_thesis_impl/coco-analyze/run_analysis.py {} {} {} {} {}"\
                                                .format(predfile, gtannfile, outdir, teamname, version)
print(cmd)

outrun_dir = latestdir(outdir)
print("Output Directory: %s\n"%outdir)


# ----------------- EVALUATE POSEFIX PREDICTIONS WITH DETAILED EVALUATION ---------------------
print("DETAILED POSEFIX EVALUATION:")
predfile = '/home/althausc/master_thesis_impl/PoseFix_RELEASE/output/result/COCO/train/23-34-53-122-3/resultfinal.json'
gtannfile = '/home/althausc/nfs/data/coco_17_medium/annotations_styletransfer/person_keypoints_train2017_stAPI.json'
outdir = '/home/althausc/master_thesis_impl/results/posedetection/posefix'
teamname = 'artinfer'
version = 1.0

cmd = "python3.6 /home/althausc/master_thesis_impl/coco-analyze/run_analysis.py {} {} {} {} {}"\
                                                .format(predfile, gtannfile, outdir, teamname, version)
print(cmd)

outrun_dir = latestdir(outdir)
print("Output Directory: %s\n"%outdir)


# ----------------- EVALUATE PREDICTION FILE WITH SHORT SUMMARY ---------------------
print("SHORT EVALUATION:")
predfile = '/home/althausc/master_thesis_impl/PoseFix_RELEASE/output/result/COCO/train/23-34-53-122-3/resultfinal.json'
gtannfile = '/home/althausc/nfs/data/coco_17_medium/annotations_styletransfer/person_keypoints_train2017_stAPI.json'
outdir = '/home/althausc/master_thesis_impl/results/posedetection/posefix' #'/home/althausc/master_thesis_impl/results/posedetection/posefix'

cmd = "python3.6 /home/althausc/master_thesis_impl/scripts/detectron2/utils/evaluateCOCOresults.py -predictions {} -gt_ann {} -outputdir {}"\
                                                .format(predfile, gtannfile)
print(cmd)

outrun_dir = latestdir(outdir)
print("Output Directory: %s\n"%outdir)


# ---------------------------- PREDICTION STATISTICS ------------------------------
print("PREDICTION STATISTICS:")
predfile = '/home/althausc/master_thesis_impl/PoseFix_RELEASE/output/result/COCO/train/23-34-53-122-3/resultfinal.json'
gtannfile = '/home/althausc/nfs/data/coco_17_medium/annotations_styletransfer/person_keypoints_train2017_stAPI.json'

cmd = "python3.6 /home/althausc/master_thesis_impl/scripts/detectron2/utils/preds_statistics.py -file {} -gtanno {}"\
                                                .format(predfile, gtannfile)
print(cmd)

outrun_dir = os.path.join(os.path.dirname(predfile), '.stats')
print("Output Directory: %s\n"%outrun_dir)

