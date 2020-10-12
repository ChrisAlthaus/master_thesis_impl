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

_PRINT_CMDS = True
_EXEC_CMDS = False

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
print("MASK-RCNN PREDICTION:")
maskrcnn_cp = '/home/althausc/master_thesis_impl/detectron2/out/checkpoints/08/07_12-40-41_all/model_0214999.pth'
gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun_G1d4-1.sh'
out_dir = '/home/althausc/master_thesis_impl/detectron2/out/art_predictions/single'
target = 'query'
topk = 20
score_tresh = 0.7

cmd = "{} python3.6 /home/althausc/master_thesis_impl/scripts/detectron2/MaskRCNN_prediction.py -model_cp {} -img {} -topk {} -score_tresh {} -target {} -vis"\
                                                                    .format(gpu_cmd, maskrcnn_cp, args.inputImg, topk, score_tresh, target)
if _PRINT_CMDS:
    print(cmd)
if _EXEC_CMDS:
    os.system(cmd)

outrun_dir = latestdir(out_dir)
print("Output Directory: %s\n"%out_dir)


# ----------------- POSEFIX PREDICTIONS ---------------------
print("POSEFIX PREDICTION:")
gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/tensorflow_srun-G1D4.sh'
model_dir = latestdir('/home/althausc/master_thesis_impl/PoseFix_RELEASE/output/model_dump/COCO')
model_epoch = 140
inputfile = os.path.join(outrun_dir,"maskrcnn_predictions.json")

#-gpu argument not used
cmd = ("{} python3.6 /home/althausc/master_thesis_impl/PoseFix_RELEASE/main/test.py --gpu 1 "+\
        "--test_epoch {} -modelfolder {} -inputs {}")\
            .format(gpu_cmd, model_epoch, model_dir, inputfile)
if _PRINT_CMDS:
    print(cmd)
if _EXEC_CMDS:
    os.system(cmd)

out_dir = '/home/althausc/master_thesis_impl/PoseFix_RELEASE/output/result/COCO'
outrun_dir = latestdir(out_dir)
print("Output Directory: %s\n"%out_dir)

#Visualize PoseFix predictions
print("VISUALIZE POSEFIX PREDICTIONS:")
ubuntu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_run.sh'
inputfile = os.path.join(outrun_dir,"resultfinal.json")
imagespath = os.path.dirname(args.inputImg) # or '/home/althausc/nfs/data/coco_17_medium/val2017_styletransfer'
outputdir = outrun_dir                      #'/home/althausc/master_thesis_impl/scripts/utils'

cmd = "{} python3.6 /home/althausc/master_thesis_impl/scripts/utils/visualizekpts.py -file {} -imagespath {} -outputdir {}"\
                                                                    .format(ubuntu_cmd, inputfile, imagespath, outputdir)
if _PRINT_CMDS:
    print(cmd)
if _EXEC_CMDS:
    os.system(cmd)
print("Output Directory: %s\n"%out_dir)


# ------------------------ GPD DESCRIPTORS ------------------------
print("CALCULATE GPD DESCRIPTORS")
_GPD_TYPES = ['JcJLdLLa_reduced', 'JLd_all']

methodgpd = _GPD_TYPES[0]
pca_on = False #True
pca_model = '/home/althausc/master_thesis_impl/posedescriptors/out/08/27_13-49-24/modelpca64.pkl'

if pca_on:
    cmd = "python3.6 /home/althausc/master_thesis_impl/scripts/pose_descriptors/geometric_pose_descriptor.py -inputFile {} -mode {} -pcamodel {} -target query"\
                                                                                                    .format(inputfile, methodgpd, pca_model)                                                                                                
else:
    cmd = "python3.6 /home/althausc/master_thesis_impl/scripts/pose_descriptors/geometric_pose_descriptor.py -inputFile {} -mode {} -target query"\
                                                                                                .format(inputfile, methodgpd)
if _PRINT_CMDS:
    print(cmd)
if _EXEC_CMDS:
    os.system(cmd)

out_dir = '/home/althausc/master_thesis_impl/posedescriptors/out/query'
outrun_dir = latestdir(out_dir)
print("Output Directory: %s\n"%out_dir)


# -------------------------- ELASTIC SEARCH -----------------------------
print("SEARCH FOR GPD IN DATABASE:")
inputfile = filewithname(outrun_dir, 'geometric_pose_descriptor')
_METHODS_SEARCH = ['COSSIM', 'DISTSUM']
_GPD_TYPES = ['JcJLdLLa_reduced', 'JLd_all']

method_search = _METHODS_SEARCH[0]
gpdtype = _GPD_TYPES[0]
evaltresh_on = True

#Querying on the database images
if method_search == 'COSSIM':
    #Not implemented so far
    #res = input("Do you want to evaluate the treshold on the gpu clustering first? [yes/no]")
    #if res == 'yes':
    #    os.system("python3.6 /home/althausc/master_thesis_impl/retrieval/elastic_search_init.py -file {} -method_search {} -evaltresh".format(inputfile, method_search))

    tresh = 0.95 #float(input("Please specify a similarity treshold for cossim result list: "))
    
    cmd = "python3.6 /home/althausc/master_thesis_impl/retrieval/elastic_search_init.py -file {} -search -method_search {} -gpd_type {} -tresh {}"\
                                                                                                .format(inputfile, method_search, gpdtype, tresh)
else:
    cmd = "python3.6 /home/althausc/master_thesis_impl/retrieval/elastic_search_init.py -file {} -search --method_search {} -gpd_type {}"\
                                                                                                .format(inputfile, method_search, gpdtype)
if _PRINT_CMDS:
    print(cmd)
if _EXEC_CMDS:
    os.system(cmd)

outdir = '/home/althausc/master_thesis_impl/retrieval/out/humanposes'
outrun_dir = latestdir(outdir)
print("Output Directory: %s\n"%out_dir)



"""
#----------------- OPTIONAL: Search with example feature files ---------------------------
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
    if method_search == 'COSSIM':
        res = input("Do you want to evaluate the treshold on the gpu clustering first? [yes/no]")
        if res == 'yes':
            os.system("python3.6 /home/althausc/master_thesis_impl/retrieval/elastic_search_init.py -file {} -method_search {} -evaltresh".format(inputfile, method_search))

        tresh = float(input("Please specify a similarity treshold for cossim result list: "))
        
        os.system("python3.6 /home/althausc/master_thesis_impl/retrieval/elastic_search_init.py -file {} -search -method_search {} -tresh {}".format(featurepath, method_search, tresh))
    else:
        os.system("python3.6 /home/althausc/master_thesis_impl/retrieval/elastic_search_init.py -file {} -search -method_search {}".format(featurepath, method_search))

"""