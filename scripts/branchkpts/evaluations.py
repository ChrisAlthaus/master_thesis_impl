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

_CREATE_FOLDERS = True

# ----------------- EVALUATE MASK-RCNN PREDICTIONS WITH DETAILED EVALUATION ---------------------
print("DETAILED MASK-RCNN EVALUATION:")
predfile = '/home/althausc/master_thesis_impl/detectron2/out/art_predictions/train/11-17_16-27-51/maskrcnn_predictions.json'
gtannfile = '/home/althausc/nfs/data/coco_17_medium/annotations_styletransfer/person_keypoints_train2017_stAPI.json'
outdir = '/home/althausc/master_thesis_impl/results/posedetection/maskrcnn'
teamname = 'artinfer'
version = 1.0

gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/sbatch_nogpu.sh'
jobname = 'detailevaluation'
logdir = '/home/althausc/master_thesis_impl/results/posedetection/maskrcnn/logs/%s'%datetime.datetime.now().strftime('%m-%d_%H-%M-%S')
if _CREATE_FOLDERS:
    os.makedirs(logdir)
logfile = os.path.join(logdir, 'log-detailed-eval.txt')

cmd = ("sbatch -w devbox4 -J {} -o {} "+ \
       "{} python3.6 /home/althausc/master_thesis_impl/coco-analyze/run_analysis.py {} {} {} {} {}")\
                                                .format(jobname, logfile, gpu_cmd, gtannfile, predfile, outdir, teamname, version)
print(cmd)
os.system(cmd)

outrun_dir = latestdir(outdir)
print("Output Directory: %s\n"%outdir)
#exit(1)

# ----------------- EVALUATE POSEFIX PREDICTIONS WITH DETAILED EVALUATION ---------------------
print("DETAILED POSEFIX EVALUATION:")
predfile = '/home/althausc/master_thesis_impl/PoseFix_RELEASE/output/result/COCO/train/23-34-53-122-3/resultfinal.json'
gtannfile = '/home/althausc/nfs/data/coco_17_medium/annotations_styletransfer/person_keypoints_train2017_stAPI.json'
outdir = '/home/althausc/master_thesis_impl/results/posedetection/posefix'
teamname = 'artinfer'
version = 1.0

gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/sbatch_nogpu.sh'
jobname = 'detailevaluation'
logdir = '/home/althausc/master_thesis_impl/results/posedetection/posefix/logs/%s'%datetime.datetime.now().strftime('%m-%d_%H-%M-%S')
if _CREATE_FOLDERS:
    os.makedirs(logdir)
logfile = os.path.join(logdir, 'log-detailed-eval.txt')

cmd = ("sbatch -w devbox4 -J {} -o {} "+ \
    "{} python3.6 /home/althausc/master_thesis_impl/coco-analyze/run_analysis.py {} {} {} {} {}")\
                                                .format(jobname, logdir, gpu_cmd, gtannfile, predfile, outdir, teamname, version)
print(cmd)

outrun_dir = latestdir(outdir)
print("Output Directory: %s\n"%outdir)


# ----------------- EVALUATE PREDICTION FILE WITH SHORT SUMMARY V1 ---------------------
print("SHORT EVALUATION (OPTION 1):")
predfile = '/home/althausc/master_thesis_impl/PoseFix_RELEASE/output/result/COCO/train/23-34-53-122-3/resultfinal.json'
gtannfile = '/home/althausc/nfs/data/coco_17_medium/annotations_styletransfer/person_keypoints_train2017_stAPI.json'
outdir = '/home/althausc/master_thesis_impl/results/posedetection/maskrcnn' #'/home/althausc/master_thesis_impl/results/posedetection/posefix'

cmd = "python3.6 /home/althausc/master_thesis_impl/scripts/detectron2/utils/evaluateCOCOresults.py -predictions {} -gt_ann {} -outputdir {}"\
                                                .format(predfile, gtannfile, outdir)
print(cmd)

outrun_dir = latestdir(outdir)
print("Output Directory: %s\n"%outdir)


# ----------------- EVALUATE PREDICTION FILE WITH SHORT SUMMARY V2 ---------------------
print("SHORT EVALUATION (OPTION 2):")
model_cp = '/home/althausc/master_thesis_impl/detectron2/out/checkpoints/11-16_16-28-06_scratch/model_final.pth'
gtannfile = '/home/althausc/nfs/data/coco_17_medium/annotations_styletransfer/person_keypoints_val2017_stAPI.json'
imagedir = '/home/althausc/nfs/data/coco_17_medium/val2017_styletransfer'
outdir = '/home/althausc/master_thesis_impl/results/posedetection/maskrcnn' #'/home/althausc/master_thesis_impl/results/posedetection/posefix'

cmd = "python3.6 /home/althausc/master_thesis_impl/scripts/detectron2/utils/evaluateCOCOresults.py -model_cp {} -imagedir {} -gt_annotations {} -outputdir {}"\
                                                .format(model_cp, imagedir, gtannfile, outdir)
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


# ----------------------------- DRAW GPD DESCRIPTORS ON IMAGES ---------------------------
print("VISUALIZING GPD DESCRIPTORS:")
predfile = '/home/althausc/master_thesis_impl/detectron2/out/art_predictions/query/11-10_16-46-44/maskrcnn_predictions.json'
gpdfile = '/home/althausc/master_thesis_impl/posedescriptors/out/query/11-10_16-47-40/geometric_pose_descriptor_c_1_mJcJLdLLa_reduced_t0.10_f0_mkpt10.json'
imagespath = '/home/althausc/master_thesis_impl/posedescriptors/out/query/12-34-5'

gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun_G1d4-1.sh'

cmd = "{} python3.6 /home/althausc/master_thesis_impl/scripts/pose_descriptors/visualizedescriptors.py  -predictionfile {} -gpdfile {} -imagespath {}"\
                                                .format(gpu_cmd, predfile, gpdfile, imagespath)
print(cmd)
 
outrun_dir = os.path.dirname(gpdfile)
print("Output Directory: %s\n"%outrun_dir)


# -------------------------- DRAW KEYPOINT PREDICTIONS ON IMAGES -----------------------------    
print("VISUALIZING KEYPOINT PREDICTIONS:")
predfile = '/home/althausc/master_thesis_impl/detectron2/out/art_predictions/query/11-10_16-46-44/maskrcnn_predictions.json'
imagespath = '/home/althausc/master_thesis_impl/posedescriptors/out/query/12-34-5'
outputdir = os.path.dirname(predfile)
isstyletransfer = False
vistresh = 0.0 # Cause Mask-RCNN prediction also has treshold filtering

gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun_G1d4-1.sh'

cmd = "{} python3.6 /home/althausc/master_thesis_impl/scripts/detectron2/utils/visualizekpts.py  -file {} -imagespath {} -outputdir {} {} -vistresh {}"\
                                                .format(gpu_cmd, predfile, imagespath, outputdir, '-transformid' if isstyletransfer else '', vistresh)
print(cmd)

print("Output Directory: %s\n"%outputdir)
