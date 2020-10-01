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

_PRINT_CMDS = True
_EXEC_CMDS = False

# ----------------- MASK-RCNN TRAIN ---------------------
print("MASK-RCNN TRAINING:")
gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun_G1d4.sh'
finetune = 'ALL'    #["RESNETF", "RESNETL", "HEADSALL", "ALL",'EVALBASELINE','FPN+HEADS','SCRATCH']
numepochs = 20
maskrcnn_cp = '/home/althausc/master_thesis_impl/detectron2/out/checkpoints/08/07_12-40-41_all/model_0214999.pth'
resume = False #True
#Additional params to setup in file: folder : BN[yes,no], LR, Weight Decay[steps& exp/no exp], Data Augmentation[Random Rotation, Flip & Crop],
#                                             Min keypoints[for filter images], Additional[MinSizeTrain, ImgPerBatch]

cmd = "{} python3.6 /home/althausc/master_thesis_impl/scripts/detectron2/MaskRCNN_train_styletransfer.py -finetune {} -numepochs {} {}"\
                                                .format(gpu_cmd, finetune, numepochs, '-resume %s'%maskrcnn_cp if resume else ' ')
if _PRINT_CMDS:
    print(cmd)
if _EXEC_CMDS:
    os.system(cmd)

out_dir = '/home/althausc/master_thesis_impl/detectron2/out/checkpoints'
outrun_dir = latestdir(out_dir)
print("Output Directory: %s\n"%out_dir)


# ----------------- MASK-RCNN PREDICTIONS ---------------------
print("MASK-RCNN PREDICTION:")
maskrcnn_cp = os.path.join(outrun_dir, 'model_0214999.pth')  #Specify model checkpoint here
gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun_G1d4.sh'
img_dir = '/home/althausc/nfs/data/coco_17_medium/train2017_styletransfer'  #Predictions used for later PoseFix training

target = 'train'
cmd = "{} python3.6 /home/althausc/master_thesis_impl/scripts/detectron2/MaskRCNN_prediction.py -model_cp {} -imgDir {} -target {} -visrandom "\
                                                                                        .format(gpu_cmd, maskrcnn_cp, img_dir, target)
if _PRINT_CMDS:
    print(cmd)
if _EXEC_CMDS:
    os.system(cmd)

out_dir = '/home/althausc/master_thesis_impl/detectron2/out/art_predictions/train'
outrun_dir = latestdir(out_dir)
print("Output Directory: %s\n"%out_dir)


# ---------- PREPROCESS PREDICTIONS ------------
print("PREPROCESS MASK-RCNN PREDICTIONS FOR POSEFIX TRAINING:")
predictionpath = os.path.join(outrun_dir, 'maskrcnn_predictions.json')
annpath = '/home/althausc/nfs/data/coco_17_medium/annotations_styletransfer/person_keypoints_train2017_stAPI.json'

cmd = "python3.6 /home/althausc/master_thesis_impl/scripts/posefix/preproc_predictions.py -predictions {} -annotations {}"\
                                                                                            .format(predictionpath, annpath)
if _PRINT_CMDS:
    print(cmd)
if _EXEC_CMDS:
    os.system(cmd)

outdir = '/home/althausc/master_thesis_impl/PoseFix_RELEASE/inputs'
outrun_dir = latestdir(out_dir)
print("Output Directory: %s\n"%out_dir)


# ----------------- POSEFIX TRAIN ---------------------
print("POSEFIX TRAINING:")
gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/tensorflow_srun-G1D4.sh'
continue_train = True #False
inputfile = os.path.join(outrun_dir,"predictions_bbox_gt.json.json")
# Annotationfile called person_keypoints_train2017_stAPI.json placed in: PoseFix_RELEASE/data/COCO/annotations
# Images for training in train2017 placed in: PoseFix_RELEASE/data/COCO/images

#-gpu argument not used
cmd = "{} python3.6 /home/althausc/master_thesis_impl/PoseFix_RELEASE/main/train.py --gpu 1 -inputpreds {} {}"\
                                        .format(gpu_cmd, inputfile, '--continue' if continue_train else ' ')
if _PRINT_CMDS:
    print(cmd)
if _EXEC_CMDS:
    os.system(cmd)

out_dir = '/home/althausc/master_thesis_impl/PoseFix_RELEASE/output/result/COCO/09'
outrun_dir = latestdir(out_dir)
print("Output Directory: %s\n"%out_dir)


# ----------------- POSEFIX PREDICTIONS ---------------------
print("POSEFIX PREDICTIONS:")
gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/tensorflow_srun-G1D3.sh'
model_epoch = 140
inputfile = os.path.join(outrun_dir,"maskrcnn_predictions.json")

#-gpu argument not used
cmd = "{} python3.6 /home/althausc/master_thesis_impl/PoseFix_RELEASE/main/test.py --gpu 1 --test_epoch {} -inputs {}".format(gpu_cmd, model_epoch, inputfile)
if _PRINT_CMDS:
    print(cmd)
if _EXEC_CMDS:
    os.system(cmd)

out_dir = '/home/althausc/master_thesis_impl/PoseFix_RELEASE/output/result/COCO'
outrun_dir = latestdir(out_dir)
print("Output Directory: %s\n"%out_dir)


# ------------------------ GPD DESCRIPTORS ------------------------
print("GENERATING GPD DESCRIPTORS:")
inputfile = '/home/althausc/master_thesis_impl/detectron2/out/art_predictions/08/18_10-29-25/result.json'
#inputfile = os.path.join(outrun_dir,"resultfinal.json")
methodgpd = 'JcJLdLLa_reduced' #['JcJLdLLa_reduced', 'JLd_all']
pca_on = True
pca_dim = 64
if pca_on:
    cmd = "python3.6 /home/althausc/master_thesis_impl/scripts/pose_descriptors/geometric_pose_descriptor.py -inputFile {} -mode {} -pca {}".format(inputfile, methodgpd, pca_dim)
else:
    cmd = "python3.6 /home/althausc/master_thesis_impl/scripts/pose_descriptors/geometric_pose_descriptor.py -inputFile {} -mode {}".format(inputfile, methodgpd)

if _PRINT_CMDS:
    print(cmd)
if _EXEC_CMDS:
    os.system(cmd)

out_dir = '/home/althausc/master_thesis_impl/posedescriptors/out/09'
outrun_dir = latestdir(out_dir)
print("Output Directory: %s\n"%out_dir)


# ------------------------- CLUSTERING ------------------------------
print("CLUSTERING (optional):")
clust_on = True
val_on = True
valmethods = ['SILHOUETTE', 'ELBOW']
ks = [10,20] #[kmin, kmax] for validation methods

inputfile = filewithname(outrun_dir, 'geometric_pose_descriptor')
print("GPD file: ",inputfile)

if clust_on:    
    for val in valmethods:
        cmd = "python3.6 /home/althausc/master_thesis_impl/scripts/pose_descriptors/clustering_descriptors.py -descriptors {} -val {} -validateks {} {}"\
                                                                                                                        .format(inputfile, val, ks[0], ks[1])
        if _PRINT_CMDS:
            print(cmd)
        if _EXEC_CMDS:
            os.system(cmd)
        #out_dir = '/home/althausc/master_thesis_impl/posedescriptors/clustering/eval'
        
    k = 100#input('Please enter the number of clusters used for gpds: ')
    cmd = "python3.6 /home/althausc/master_thesis_impl/scripts/pose_descriptors/clustering_descriptors.py -descriptors {} -buildk {}".format(inputfile, k)
    if _PRINT_CMDS:
        print(cmd)
    if _EXEC_CMDS:
        os.system(cmd)

    out_dir = '/home/althausc/master_thesis_impl/posedescriptors/clustering/out'
    outrun_dir = latestdir(out_dir)
else:
    out_dir = '/home/althausc/master_thesis_impl/posedescriptors/out'
    outrun_dir = latestdir(out_dir)

print("Output Directory: %s\n"%out_dir) 


# -------------------------- ELASTIC Database INSERT -----------------------------
print("ELASTIC DATABASE INSERT:")
if clust_on:
    inputfile = filewithname(outrun_dir, 'codebook_mapping.txt')
    print("Clustering file: ",inputfile)

    method_insert = 'CLUSTER' #['CLUSTER', 'RAW'] 
    cmd = "python3.6 /home/althausc/master_thesis_impl/retrieval/elastic_search_init.py -file {} -insert -method_ins {} -gpd_type {}"\
                                                                                            .format(inputfile, method_insert, methodgpd)
    if _PRINT_CMDS:
        print(cmd)
    if _EXEC_CMDS:
        os.system(cmd)
else:
    inputfile = filewithname(outrun_dir, 'geometric_pose_descriptor')
    print("GPD file: ",inputfile)

    method_insert = 'RAW' #['CLUSTER', 'RAW'] 
    cmd = "python3.6 /home/althausc/master_thesis_impl/retrieval/elastic_search_init.py -file {} -insert -method_ins {} -gpd_type {}"\
                                                                                            .format(inputfile, method_insert, methodgpd)
    if _PRINT_CMDS:
        print(cmd)
    if _EXEC_CMDS:
        os.system(cmd)

outdir = '/home/althausc/master_thesis_impl/retrieval/out/configs'
print("Output Directory: %s\n"%out_dir)




