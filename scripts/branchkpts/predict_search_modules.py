import os
import json
import numpy as np
import math
import pickle
import datetime
import time
import logging
import itertools
from PIL import Image


def latestdir(dir):
    diritems = [os.path.join(dir, d) for d in os.listdir(dir)]
    all_subdirs = [d for d in diritems if os.path.isdir(d)]
    return max(all_subdirs, key=os.path.getmtime)

def filewithname(dir, searchstr):
    for item in os.listdir(dir):
        if os.path.isfile(os.path.join(dir,item)) and searchstr in item:
            return os.path.join(dir,item)
    return None

logpath = '/home/althausc/master_thesis_impl/results/logs/jupyter-notebook'

def test():
    imgpath = '/home/althausc/nfs/data/coco_17_medium/val2017_styletransfer/000000000785_050351.jpg'
    predict(imgpath)

def predict(imgpath):
    # ----------------- MASK-RCNN PREDICTIONS ---------------------
    print("MASK-RCNN PREDICTION:")
    maskrcnn_cp = '/home/althausc/master_thesis_impl/detectron2/out/checkpoints/08/07_12-40-41_all/model_0214999.pth'
    gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun_G1d3.sh'
    out_dir = '/home/althausc/master_thesis_impl/detectron2/out/art_predictions/09'
    logfile = os.path.join(logpath, '1-maskrcnn.txt')

    if os.system("{} python3.6 /home/althausc/master_thesis_impl/scripts/detectron2/MaskRCNN_prediction.py \-model_cp {} -img {} -vis > {}"\
                                                                                                .format(gpu_cmd, maskrcnn_cp, imgpath, logfile)):
        raise RuntimeError('Mask RCNN Prediction failed.')
    

    outrun_dir = latestdir(out_dir)
    print("\n\n")

    # ----------------- POSEFIX PREDICTIONS ---------------------
    print("POSEFIX PREDICTION:")
    gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/tensorflow_srun-G1D3.sh'
    model_dir = latestdir('/home/althausc/master_thesis_impl/PoseFix_RELEASE/output/model_dump/COCO')
    model_epoch = 140
    inputfile = os.path.join(outrun_dir,"maskrcnn_predictions.json")
    logfile = os.path.join(logpath, '2-posefix.txt')

    #-gpu argument not used
    if os.system("{} python3.6 /home/althausc/master_thesis_impl/PoseFix_RELEASE/main/test.py --gpu 1 --test_epoch {} -modelfolder {} -inputs {} > {}"\
                                                                                            .format(gpu_cmd, model_epoch, model_dir, inputfile, logfile)):
        raise RuntimeError('PoseFix Prediction failed.')

    out_dir = '/home/althausc/master_thesis_impl/PoseFix_RELEASE/output/result/COCO/09'
    outrun_dir = latestdir(out_dir)
    print("\n\n")

    #Visualize PoseFix predictions
    print("VISUALIZE POSEFIX PREDICTIONS:")
    ubuntu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_run.sh'
    inputfile = os.path.join(outrun_dir,"resultfinal.json")
    imagespath = os.path.dirname(imgpath) #'/home/althausc/nfs/data/coco_17_medium/val2017_styletransfer'
    #Save in same dir as image of imgpath
    outputdir = os.path.dirname(imgpath) 
    logfile = os.path.join(logpath, '3-visualize.txt')

    if os.system("{} python3.6 /home/althausc/master_thesis_impl/scripts/utils/visualizekpts.py -file {} -imagespath {} -outputdir {} > {}"\
                                                                                            .format(ubuntu_cmd, inputfile, imagespath, outputdir, logfile)):
        raise RuntimeError('Visualize prediction failed.')
    print("\n\n")

    annpath = inputfile
    return annpath

def transform_to_gpd(annpath, methodgpd, pca_on=False, pca_model=None):
    # ------------------------ GPD DESCRIPTORS ------------------------
    print("CALCULATE GPD DESCRIPTORS")
    #methodgpd = 0 #['JcJLdLLa_reduced', 'JLd_all']
    #pca_on = False #True
    #pca_model = '/home/althausc/master_thesis_impl/posedescriptors/out/08/27_13-49-24/modelpca64.pkl'
    
    if pca_on:
        if pca_model is None:
            raise ValueError("Please specify a pca model file for feature reduction.")
        os.system("python3.6 /home/althausc/master_thesis_impl/scripts/pose_descriptors/geometric_pose_descriptor.py -inputFile {} -mode {} -pcamodel {}".format(annpath, methodgpd, pca_model))
    else:
        os.system("python3.6 /home/althausc/master_thesis_impl/scripts/pose_descriptors/geometric_pose_descriptor.py -inputFile {} -mode {}".format(annpath, methodgpd))

    out_dir = '/home/althausc/master_thesis_impl/posedescriptors/out/09'
    outrun_dir = latestdir(out_dir)

    print("\n\n")
    gpdfile = filewithname(outrun_dir, 'geometric_pose_descriptor')
    return gpdfile

def search(gpdfile, method_search, tresh):
    # -------------------------- ELASTIC SEARCH -----------------------------
    print("SEARCH FOR GPD IN DATABASE:")

    inputfile = gpdfile
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

    outrun_dir = latestdir('/home/althausc/master_thesis_impl/retrieval/out')
    rankingfile = os.path.join(outrun_dir, 'result-ranking.json')

    return rankingfile 


def getImgs(rankingfile):
    print("Reading from file: ",rankingfile)
    with open (rankingfile, "r") as f:
        json_data = json.load(f)

    imagedir = json_data['imagedir']
    del json_data['imagedir']

    rankedlist = sorted(json_data.items(), key= lambda x: int(x[0])) 
    imgs = []
    scores = []
    for item in rankedlist:
        #imgs.append(Image.open(item[1]['filepath']))
        imgs.append(Image.open(os.path.join(imagedir,item[1]['filepath'])))
        scores.append(item[1]['relscore'])
    
    return imgs, scores
        
def treshIndex(tresh, results):
    with open (results, "r") as f:
        json_data = json.load(f)

    imagedir = json_data['imagedir']
    del json_data['imagedir']

    rankedlist = sorted(list(json_data.items()), key= lambda x: int(x[0])) 
    
    imgs = []
    k = 0
    for item in rankedlist:
        print(item)
        if item[1]['relscore']< tresh:
            break
        else:
            k = k + 1
    return k

#import ipyplot
#ipyplot.plot_images(imgs, max_images=10, img_width=150)
if __name__=="__main__":
   #test()
   print(getImgs('/home/althausc/master_thesis_impl/retrieval/out/09/02_13-16-41/result-ranking.json'))