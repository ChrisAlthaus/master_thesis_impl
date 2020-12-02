
import numpy as np
import datetime
import os
import time
from PIL import Image
import copy
import json

import argparse


parser = argparse.ArgumentParser()
#Calculate predictions on-the-fly
parser.add_argument('-model_cp', help='Path to the model checkpoint (no prediction file needed).')
parser.add_argument('-image', help='Path to the image which should be predicted.')


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
# ---------------------------------------- PREDICTION ----------------------------------------
print("MASK-RCNN PREDICTION:")
predictionfile = ''
gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun_G1d3-1.sh'
topk = 1
score_tresh = 0.9 #For evaluation use all predictions
styletransfered = False

cmd = "{} python3.6 /home/althausc/master_thesis_impl/scripts/detectron2/MaskRCNN_prediction.py -model_cp {} -img {} -topk {} -score_tresh {} -target eval {} -vis"\
                                                .format(gpu_cmd, args.model_cp, args.image, topk, score_tresh, '-styletransfered' if styletransfered else ' ')
print(cmd)
os.system(cmd)
outputdir = latestdir('/home/althausc/master_thesis_impl/detectron2/out/art_predictions/eval')
print("Output directory: ", outputdir)
predictionfile = os.path.join(outputdir, "maskrcnn_predictions.json")
#'/home/althausc/master_thesis_impl/detectron2/out/art_predictions/query/11-27_18-56-33/maskrcnn_predictions.json' #os.path.join(latestdir(outputdir), 'maskrcnn_predictions.json')"""


# -------------------------------- GENERATING IMAGE PATCHES --------------------------------------
print("GENERATING IMAGE PATCHES:")
outputdir = '/home/althausc/master_thesis_impl/detectron2/out/art_predictions/eval/12-01_10-30-06'
predictionfile = '/home/althausc/master_thesis_impl/detectron2/out/art_predictions/eval/12-01_10-30-06/maskrcnn_predictions.json'

imgname = os.path.splitext(os.path.basename(args.image))[0]

img = Image.open(args.image) 
width, height = img.size 

patchsizes = [(0.5*width, 0.5*height),
              (0.6*width, 0.6*height),
              (0.7*width, 0.7*height),
              (0.8*width, 0.8*height),
              (0.9*width, 0.9*height),
              (1*width, 1*height)]
translations = [(0.1*width, 0.1*height),
                (0.1*width, 0.1*height),
                (0.1*width, 0.1*height),
                (0.1*width, 0.1*height),
                (0.1*width, 0.1*height),
                (0.1*width, 0.1*height)]

data = [] #to now only the first prediction supported
print("Reading from file: ", predictionfile)
with open (predictionfile, "r") as f:
    data = json.load(f)

pred = data[0]
assert width == pred['image_size'][1]
assert height == pred['image_size'][0]

dataout = [] #cropped/reduced predictions for each generated image

def getprediction(fullprediction, newx, newy, newwidth, newheight, id):
    newprediction = copy.deepcopy(fullprediction)
    del newprediction['bbox']
    #del newprediction['category_id']
    newprediction['image_id'] = id
    newprediction['image_size'] = [newheight, newwidth]
    #print(newx, newy, newx + newwidth,  newy + newheight)
    c=0
    for n in range(0, len(newprediction['keypoints']), 3):
        kx =  newprediction['keypoints'][n+0]
        ky =  newprediction['keypoints'][n+1]
        
        if kx < newx or kx > newx + newwidth or ky < newy or ky > newy + newheight:
            newprediction['keypoints'][n+2] = 0.0

        newprediction['keypoints'][n+0] = newprediction['keypoints'][n+0] - newx
        newprediction['keypoints'][n+1] = newprediction['keypoints'][n+1] - newy

        c = c + 1
    assert c == 17
    return newprediction

id = 0
numerate = [1,1]
for k, patchsize in enumerate(patchsizes):
    t = [0.0, 0.0]
    tadd = translations[k]
    while t[1] + patchsize[1] <= height:
        while t[0] + patchsize[0] <= width:
            imgc = img.crop((t[0], t[1], t[0] + patchsize[0], t[1] + patchsize[1])) 
            #imgc.save(os.path.join(outputdir, imgname + '-{}_{}{}.jpg'.format(k,numerate[0], numerate[1])))
            imgc.save(os.path.join(outputdir, '.visimages', '{}.jpg'.format(id)))

            newprediction = getprediction(pred, t[0], t[1], patchsize[0], patchsize[1], id)
            dataout.append(newprediction)
            #print(pred)
            #print(newprediction)
            #print('-' * 20)
            t[0] = t[0] + tadd[0]
            numerate[1] = numerate[1] + 1
            id = id + 1

        t[0] = 0.0
        numerate[1] = 0 
        t[1] = t[1] + tadd[1]  
        numerate[0] = numerate[0] + 1
    numerate[0], numerate[1] = 0.0, 0.0

print("Output directory: ", outputdir)
fileout = os.path.join(outputdir, "maskrcnn_predictions_patches.json")
with open(fileout, 'w') as f:
    json.dump(dataout, f, separators=(', ', ': '))


# ---------------------------------------- VISUALIZE KEYPOINTS ----------------------------------------
print("VISUALIZE KEYPOINTS FROM REDUCED PREDICTIONS:")
gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun_G1d3-1.sh'
#os.makedirs(os.path.join(outputdir, '.evalimages'))

cmd = "{} python3.6 /home/althausc/master_thesis_impl/scripts/detectron2/utils/visualizekpts.py -file {} -imagespath {} -outputdir {}"\
                                                .format(gpu_cmd, fileout, os.path.join(outputdir, ".visimages"), os.path.join(outputdir, '.evalimages'))
print(cmd)
os.system(cmd)

imgdir = os.path.join(outputdir, '.visimages')
print("Output directory: ", imgdir)


# ---------------------------------------- CALCULATE GPD DESCRIPTORS ----------------------------------------
print("CALCULATE GPD DESCRIPTORS ...")
methodgpd = 'JcJLdLLa_reduced' #['JcJLdLLa_reduced', 'JLd_all']
target = 'eval'

cmd = "python3.6 /home/althausc/master_thesis_impl/scripts/pose_descriptors/geometric_pose_descriptor.py -inputFile {} -mode {} -target {}"\
                                                                                            .format(fileout, methodgpd, target)
print(cmd)
os.system(cmd)

out_dir = '/home/althausc/master_thesis_impl/posedescriptors/out/eval'
outrun_dir = latestdir(out_dir)
print("Output directory: ", outrun_dir)


# ---------------------------------------- PRINT ELASTICSEARCH INSERT ----------------------------------------
methodins = 'RAW' #['CLUSTER', 'RAW']
gpdfile = filewithname(outrun_dir, 'geometric_pose_descriptor')
cmd = "python3.6 /home/althausc/master_thesis_impl/retrieval/elastic_search_init.py -file {} -insert -method_ins {} -imgdir {} -gpd_type {}"\
                                                                                            .format(gpdfile, methodins, imgdir, methodgpd)
print(cmd)

# ---------------------------------------- CREATE QUERY DESCRIPTOR ----------------------------------------
#Can be used by jupyter notebook
methodins = 'RAW' #['CLUSTER', 'RAW']
imgid = int(input("Image id for new gpd descriptor: "))
with open (gpdfile, "r") as f:
    data = f.read()
data = eval(data)
descriptor = []
for gpd in data:
    if gpd['image_id'] == imgid:
        descriptor.append(gpd)
assert len(descriptor) != 0

with open(os.path.join(outrun_dir, 'gpd_evalsingle.json'), 'w') as f:
    print("Writing to folder: ",outrun_dir)
    json.dump(descriptor, f)