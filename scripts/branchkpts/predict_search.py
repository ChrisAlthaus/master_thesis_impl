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
parser.add_argument('-inputImg',required=True,
                    help='Image for which to infer best k matching images.')
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

# ----------------- MASK-RCNN PREDICTIONS ---------------------
maskrcnn_cp = '/home/althausc/master_thesis_impl/detectron2/out/checkpoints/08/07_12-40-41_all/model_0214999.pth'
gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun_G1d3.sh'
out_dir = '/home/althausc/master_thesis_impl/detectron2/out/art_predictions/09'

os.system("{} python3.6 /home/althausc/master_thesis_impl/scripts/detectron2/MaskRCNN_prediction.py -model_cp {} -img {} -vis".format(gpu_cmd, maskrcnn_cp, args.inputImg))

outrun_dir = latestdir(out_dir)
with open(os.path.join(outrun_dir,"maskrcnn_predictions.json"), 'r') as f:
    json_data = json.load(f)

print(json_data)

# ----------------- POSEFIX PREDICTIONS ---------------------
gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/tensorflow_srun-G1D3.sh'
model_epoch = 25
inputfile = os.path.join(outrun_dir,"maskrcnn_predictions.json")

#-gpu argument not used
os.system("{} python3.6 /home/althausc/master_thesis_impl/PoseFix_RELEASE/main/test.py --gpu 1 --test_epoch {} -inputs {}".format(gpu_cmd, model_epoch, inputfile))

out_dir = '/home/althausc/master_thesis_impl/PoseFix_RELEASE/output/result/COCO/09'
outrun_dir = latestdir(out_dir)

#Visualize PoseFix predictions
ubuntu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_run.sh'
inputfile = os.path.join(outrun_dir,"resultfinal.json")
imagespath = '/home/althausc/nfs/data/coco_17_medium/val2017_styletransfer'
outputdir = '/home/althausc/master_thesis_impl/scripts/utils'

os.system("{} python3.6 /home/althausc/master_thesis_impl/scripts/utils/visualizekpts.py -file {} -imagespath {} -outputdir {}".format(ubuntu_cmd, inputfile, imagespath, outputdir))


# ------------------------ GPD DESCRIPTORS ------------------------

methodgpd = 1 #['JcJLdLLa_reduced', 'JLd_all']
pca_on = True
pca_dim = 64
if pca_on:
    os.system("python3.6 /home/althausc/master_thesis_impl/scripts/pose_descriptors/geometric_pose_descriptor.py -inputFile {} -mode {} -pca {}".format(inputfile, methodgpd, pca_dim))
else:
    os.system("python3.6 /home/althausc/master_thesis_impl/scripts/pose_descriptors/geometric_pose_descriptor.py -inputFile {} -mode {}".format(inputfile, methodgpd))


# ------------------------- CLUSTERING ------------------------------
clust_on = True
val_on = True
valmethods = ['SILHOUETTE', 'ELBOW']

out_dir = '/home/althausc/master_thesis_impl/posedescriptors/out/09'
outrun_dir = latestdir(out_dir)
inputfile = filewithname(outrun_dir, 'geometric_pose_descriptor')
print("GPD file: ",inputfile)

if clust_on:    
    for val in valmethods:
        os.system("python3.6 clustering_descriptors.py -descriptors {} -val {}".format(inputfile, val))

    k = input('Please enter the number of clusters used for gpds: ')
    os.system("python3.6 clustering_descriptors.py -descriptors {} -buildk {}".format(inputfile, k))


# -------------------------- ELASTIC SEARCH -----------------------------

if clust_on:
    out_dir = '/home/althausc/master_thesis_impl/posedescriptors/clustering/out/09'
    outrun_dir = latestdir(out_dir)

    inputfile = filewithname(outrun_dir, 'codebook_mapping.txt')
    print("Clustering file: ",inputfile)

method = 'COSSIM' #['COSSIM', 'DISTSUM']
evaltresh_on = True
os.system("python3.6 /home/althausc/master_thesis_impl/retrieval/elastic_search_init.py -file {} -insert -method {}".format(inputfile, method))

#Search with examples
while(True):
    featurepath = input("You can search for images in the database by specifying a file with featurevectors: ")
    if not os.path.exists(featurepath):
        print("No valid file path.")
        continue

    #Transform feature vector in gpd descriptor 
    if pca_on:
        pca_model = input("Please specify the path to a pca model for feature dim reduction: ")
        os.system("python3.6 /home/althausc/master_thesis_impl/scripts/pose_descriptors/geometric_pose_descriptor.py -inputFile {} -mode {} -pcamodel {}".format(inputfile, methodgpd, pca_model))
    else:
        os.system("python3.6 /home/althausc/master_thesis_impl/scripts/pose_descriptors/geometric_pose_descriptor.py -inputFile {} -mode {}".format(inputfile, methodgpd))

    out_dir = '/home/althausc/master_thesis_impl/posedescriptors/out/09'
    outrun_dir = latestdir(out_dir)
    inputfile = filewithname(outrun_dir, 'geometric_pose_descriptor')
    print("GPD file: ",inputfile)

    #Querying on the database images
    if method == 'COSSIM':
        res = input("Do you want to evaluate the treshold on the gpu clustering first? [yes/no]")
        if res == 'yes':
            os.system("python3.6 /home/althausc/master_thesis_impl/retrieval/elastic_search_init.py -file {} -method {} -evaltresh".format(inputfile, method))

        tresh = float(input("Please specify a similarity treshold for cossim result list: "))
        
        os.system("python3.6 /home/althausc/master_thesis_impl/retrieval/elastic_search_init.py -file {} -search -method {} -tresh".format(featurepath, method, tresh))
    else:
        os.system("python3.6 /home/althausc/master_thesis_impl/retrieval/elastic_search_init.py -file {} -search -method {}".format(featurepath, method))

