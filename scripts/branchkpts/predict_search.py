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
print("MASK-RCNN PREDICTION:")
maskrcnn_cp = '/home/althausc/master_thesis_impl/detectron2/out/checkpoints/08/07_12-40-41_all/model_0214999.pth'
gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun_G1d3.sh'
out_dir = '/home/althausc/master_thesis_impl/detectron2/out/art_predictions/09'

os.system("{} python3.6 /home/althausc/master_thesis_impl/scripts/detectron2/MaskRCNN_prediction.py -model_cp {} -img {} -vis".format(gpu_cmd, maskrcnn_cp, args.inputImg))

outrun_dir = latestdir(out_dir)
with open(os.path.join(outrun_dir,"maskrcnn_predictions.json"), 'r') as f:
    json_data = json.load(f)

print("Prediction output: ",json_data)
print("\n\n")

# ----------------- POSEFIX PREDICTIONS ---------------------
print("POSEFIX PREDICTION:")
gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/tensorflow_srun-G1D3.sh'
model_dir = latestdir('/home/althausc/master_thesis_impl/PoseFix_RELEASE/output/model_dump/COCO')
model_epoch = 140
inputfile = os.path.join(outrun_dir,"maskrcnn_predictions.json")

#-gpu argument not used
os.system("{} python3.6 /home/althausc/master_thesis_impl/PoseFix_RELEASE/main/test.py --gpu 1 --test_epoch {} -modelfolder {} -inputs {}".format(gpu_cmd, model_epoch, model_dir, inputfile))

out_dir = '/home/althausc/master_thesis_impl/PoseFix_RELEASE/output/result/COCO/09'
outrun_dir = latestdir(out_dir)
print("\n\n")

#Visualize PoseFix predictions
print("VISUALIZE POSEFIX PREDICTIONS:")
ubuntu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_run.sh'
inputfile = os.path.join(outrun_dir,"resultfinal.json")
imagespath = os.path.dirname(args.inputImg) #'/home/althausc/nfs/data/coco_17_medium/val2017_styletransfer'
outputdir = outrun_dir                      #'/home/althausc/master_thesis_impl/scripts/utils'

os.system("{} python3.6 /home/althausc/master_thesis_impl/scripts/utils/visualizekpts.py -file {} -imagespath {} -outputdir {}".format(ubuntu_cmd, inputfile, imagespath, outputdir))
print("\n\n")


# ------------------------ GPD DESCRIPTORS ------------------------
print("CALCULATE GPD DESCRIPTORS")
methodgpd = 0 #['JcJLdLLa_reduced', 'JLd_all']
pca_on = False #True
pca_model = '/home/althausc/master_thesis_impl/posedescriptors/out/08/27_13-49-24/modelpca64.pkl'

if pca_on:
    os.system("python3.6 /home/althausc/master_thesis_impl/scripts/pose_descriptors/geometric_pose_descriptor.py -inputFile {} -mode {} -pcamodel {}".format(inputfile, methodgpd, pca_model))
else:
    os.system("python3.6 /home/althausc/master_thesis_impl/scripts/pose_descriptors/geometric_pose_descriptor.py -inputFile {} -mode {}".format(inputfile, methodgpd))

out_dir = '/home/althausc/master_thesis_impl/posedescriptors/out/09'
outrun_dir = latestdir(out_dir)
print("\n\n")


# -------------------------- ELASTIC SEARCH -----------------------------
print("SEARCH FOR GPD IN DATABASE:")

inputfile = filewithname(outrun_dir, 'geometric_pose_descriptor')
print("GPD file: ",inputfile)

method_search = 'COSSIM' #['COSSIM', 'DISTSUM']
evaltresh_on = True

#Querying on the database images
if method_search == 'COSSIM':
    #Not implemented so far
    #res = input("Do you want to evaluate the treshold on the gpu clustering first? [yes/no]")
    #if res == 'yes':
    #    os.system("python3.6 /home/althausc/master_thesis_impl/retrieval/elastic_search_init.py -file {} -method_search {} -evaltresh".format(inputfile, method_search))

    tresh = float(input("Please specify a similarity treshold for cossim result list: "))
    
    os.system("python3.6 /home/althausc/master_thesis_impl/retrieval/elastic_search_init.py -file {} -search -method_search {} -tresh {}".format(inputfile, method_search, tresh))
else:
    os.system("python3.6 /home/althausc/master_thesis_impl/retrieval/elastic_search_init.py -file {} -search --method_search {}".format(inputfile, method_search))
print('\n\n')


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

