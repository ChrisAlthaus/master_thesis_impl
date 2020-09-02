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
"""
# ----------------- MASK-RCNN TRAIN ---------------------
#maskrcnn_cp = '/home/althausc/master_thesis_impl/detectron2/out/checkpoints/08/07_12-40-41_all/model_0214999.pth'
gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun_G1d3.sh'
out_dir = '/home/althausc/master_thesis_impl/detectron2/out/checkpoints/09'
finetune = 'ALL'    #["RESNETF", "RESNETL", "HEADSALL", "ALL",'EVALBASELINE','FPN+HEADS','SCRATCH']
numepochs = 20
#Additional params to setup in file: folder : BN[yes,no], LR, Weight Decay[steps& exp/no exp], Data Augmentation[Random Rotation, Flip & Crop],
#                                             Min keypoints[for filter images], Additional[MinSizeTrain, ImgPerBatch]

os.system("{} python3.6 /home/althausc/master_thesis_impl/scripts/detectron2/MaskRCNN_train_styletransfer.py -finetune {} -numepochs {}".format(gpu_cmd, finetune, numepochs))
outrun_dir = latestdir(out_dir)

# ----------------- MASK-RCNN PREDICTIONS ---------------------
#Specify model checkpoint here
maskrcnn_cp = os.path.join(outrun_dir, 'model_0214999.pth')
gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun_G1d3.sh'
img_dir = '/home/althausc/nfs/data/coco_17_medium/train2017_styletransfer'
out_dir = '/home/althausc/master_thesis_impl/detectron2/out/art_predictions/09'

os.system("{} python3.6 /home/althausc/master_thesis_impl/scripts/detectron2/MaskRCNN_prediction.py -model_cp {} -imgDir {} -visrandom".format(gpu_cmd, maskrcnn_cp, img_dir))
outrun_dir = latestdir(out_dir)



# ----------------- POSEFIX TRAIN ---------------------
gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/tensorflow_srun-G1D3.sh'
model_epoch = 140
inputfile = os.path.join(outrun_dir,"maskrcnn_predictions.json")
# Annotationfile called person_keypoints_train2017_stAPI.json placed in: PoseFix_RELEASE/data/COCO/annotations
# Images for training in train2017 placed in: PoseFix_RELEASE/data/COCO/images

#-gpu argument not used
os.system("{} python3.6 /home/althausc/master_thesis_impl/PoseFix_RELEASE/main/train.py --gpu 1 --test_epoch {} -inputpreds {}".format(gpu_cmd, model_epoch, inputfile))

out_dir = '/home/althausc/master_thesis_impl/PoseFix_RELEASE/output/result/COCO/09'
outrun_dir = latestdir(out_dir)

# ----------------- POSEFIX PREDICTIONS ---------------------
gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/tensorflow_srun-G1D3.sh'
model_epoch = 140
inputfile = os.path.join(outrun_dir,"maskrcnn_predictions.json")

#-gpu argument not used
os.system("{} python3.6 /home/althausc/master_thesis_impl/PoseFix_RELEASE/main/test.py --gpu 1 --test_epoch {} -inputs {}".format(gpu_cmd, model_epoch, inputfile))

out_dir = '/home/althausc/master_thesis_impl/PoseFix_RELEASE/output/result/COCO/09'
outrun_dir = latestdir(out_dir)
"""
#Results of PoseFix Predictions
out_dir = '/home/althausc/master_thesis_impl/PoseFix_RELEASE/output/result/COCO/09'
outrun_dir = latestdir(out_dir)

# ------------------------ GPD DESCRIPTORS ------------------------
inputfile = '/home/althausc/master_thesis_impl/detectron2/out/art_predictions/08/18_10-29-25/result.json'
#inputfile = os.path.join(outrun_dir,"resultfinal.json")
methodgpd = 1 #['JcJLdLLa_reduced', 'JLd_all']
pca_on = True
pca_dim = 64
if pca_on:
    os.system("python3.6 /home/althausc/master_thesis_impl/scripts/pose_descriptors/geometric_pose_descriptor.py -inputFile {} -mode {} -pca {}".format(inputfile, methodgpd, pca_dim))
else:
    os.system("python3.6 /home/althausc/master_thesis_impl/scripts/pose_descriptors/geometric_pose_descriptor.py -inputFile {} -mode {}".format(inputfile, methodgpd))

out_dir = '/home/althausc/master_thesis_impl/posedescriptors/out/09'
outrun_dir = latestdir(out_dir)


# ------------------------- CLUSTERING ------------------------------
clust_on = True
val_on = True
valmethods = ['SILHOUETTE', 'ELBOW']
ks = [10,20] #[kmin, kmax] for validation methods

inputfile = filewithname(outrun_dir, 'geometric_pose_descriptor')
print("GPD file: ",inputfile)

if clust_on:    
    for val in valmethods:
        os.system("python3.6 /home/althausc/master_thesis_impl/scripts/pose_descriptors/clustering_descriptors.py -descriptors {} -val {} -validateks {} {}"\
                                                                                                                        .format(inputfile, val, ks[0], ks[1]))

    k = input('Please enter the number of clusters used for gpds: ')
    os.system("python3.6 /home/althausc/master_thesis_impl/scripts/pose_descriptors/clustering_descriptors.py -descriptors {} -buildk {}".format(inputfile, k))


# -------------------------- ELASTIC Database INSERT -----------------------------

if clust_on:
    out_dir = '/home/althausc/master_thesis_impl/posedescriptors/clustering/out/09'
    outrun_dir = latestdir(out_dir)
    inputfile = filewithname(outrun_dir, 'codebook_mapping.txt')
    print("Clustering file: ",inputfile)
else:
    out_dir = '/home/althausc/master_thesis_impl/posedescriptors/out/09'
    outrun_dir = latestdir(out_dir)
    inputfile = filewithname(outrun_dir, 'geometric_pose_descriptor')
    print("GPD file: ",inputfile)

method_insert = 'CLUSTER' #['CLUSTER', 'RAW'] 
os.system("python3.6 /home/althausc/master_thesis_impl/retrieval/elastic_search_init.py -file {} -insert -method_ins {}".format(inputfile, method_insert))



