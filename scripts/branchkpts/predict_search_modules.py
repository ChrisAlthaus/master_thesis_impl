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
import cv2


def latestdir(dir):
    diritems = [os.path.join(dir, d) for d in os.listdir(dir)]
    all_subdirs = [d for d in diritems if os.path.isdir(d)]
    return max(all_subdirs, key=os.path.getmtime)

def filewithname(dir, searchstr):
    for item in os.listdir(dir):
        if os.path.isfile(os.path.join(dir,item)) and searchstr in item:
            return os.path.join(dir,item)
    return None

logpath = '/home/althausc/master_thesis_impl/results/logs/jupyter-notebook/kptbranch'
if not os.path.exists(logpath):
    os.makedirs(logpath)
    print("Successfully created output directory: ", logpath)

def test():
    #imgpath = '/home/althausc/nfs/data/coco_17_medium/val2017_styletransfer/000000000785_050351.jpg'
    #predict(imgpath)
    import os
    print(os.getcwd())

def is_styletranfered_img(imgpath):
    imgname = os.path.splitext(os.path.basename(imgpath))[0]
    imgname = imgname.replace("_","")
    
    if imgname.isdigit() and len(imgname)<=18:
        return True
    else:
        return False


def predict(imgpath):
    # ----------------- MASK-RCNN PREDICTIONS ---------------------
    print("MASK-RCNN PREDICTION ...")
    maskrcnn_cp = '/home/althausc/master_thesis_impl/detectron2/out/checkpoints/08/07_12-40-41_all/model_0214999.pth'
    gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun_G1d4.sh'
    out_dir = '/home/althausc/master_thesis_impl/detectron2/out/art_predictions/09'
    transform_arg = "-tranformid" if is_styletranfered_img(imgpath) else ""
    logfile = os.path.join(logpath, '1-maskrcnn.txt')

    #print("{} python3.6 /home/althausc/master_thesis_impl/scripts/detectron2/MaskRCNN_prediction.py \-model_cp {} -img {} -vis > {}"\
    #                                                                                            .format(gpu_cmd, maskrcnn_cp, imgpath, logfile))

    if os.system("{} python3.6 /home/althausc/master_thesis_impl/scripts/detectron2/MaskRCNN_prediction.py \-model_cp {} -img {} {} -vis &> {}"\
                                                                                        .format(gpu_cmd, maskrcnn_cp, imgpath, transform_arg, logfile)):
        raise RuntimeError('Mask RCNN Prediction failed.')
    

    outrun_dir = latestdir(out_dir)
    print("MASK-RCNN PREDICTION DONE.")

    # ----------------- POSEFIX PREDICTIONS ---------------------
    print("POSEFIX PREDICTION ...")
    gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/tensorflow_srun-G1D4.sh'
    model_dir = latestdir('/home/althausc/master_thesis_impl/PoseFix_RELEASE/output/model_dump/COCO')
    model_epoch = 140
    inputfile = os.path.join(outrun_dir,"maskrcnn_predictions.json")
    logfile = os.path.join(logpath, '2-posefix.txt')

    #print("{} python3.6 /home/althausc/master_thesis_impl/PoseFix_RELEASE/main/test.py --gpu 1 --test_epoch {} -modelfolder {} -inputs {} > {}"\
    #                                                                                        .format(gpu_cmd, model_epoch, model_dir, inputfile, logfile))
    
    #-gpu argument not used
    if os.system("{} python3.6 /home/althausc/master_thesis_impl/PoseFix_RELEASE/main/test.py --gpu 1 --test_epoch {} -modelfolder {} -inputs {} {} &> {}"\
                                                                                    .format(gpu_cmd, model_epoch, model_dir, inputfile, transform_arg, logfile)):
        raise RuntimeError('PoseFix Prediction failed.')

    out_dir = '/home/althausc/master_thesis_impl/PoseFix_RELEASE/output/result/COCO/09'
    outrun_dir = latestdir(out_dir)
    print("POSEFIX PREDICTION DONE.")

    #Visualize PoseFix predictions
    print("VISUALIZE POSEFIX PREDICTIONS ...")
    ubuntu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_run.sh'
    inputfile = os.path.join(outrun_dir,"resultfinal.json")
    imagespath = os.path.dirname(imgpath) #'/home/althausc/nfs/data/coco_17_medium/val2017_styletransfer'
    #Save visualization image in dir from which the script is started
    outputdir = os.path.dirname(os.path.realpath(__file__))
    logfile = os.path.join(logpath, '3-visualize.txt')

   
    if os.system("{} python3.6 /home/althausc/master_thesis_impl/scripts/utils/visualizekpts.py -file {} -imagespath {} -outputdir {} {} &> {}"\
                                                                                .format(ubuntu_cmd, inputfile, imagespath, outputdir, transform_arg, logfile)):
        raise RuntimeError('Visualize prediction failed.')
    print("VISUALIZE POSEFIX PREDICTIONS DONE.")

    annpath = inputfile
    return annpath

def transform_to_gpd(annpath, methodgpd, pca_on=False, pca_model=None):
    # ------------------------ GPD DESCRIPTORS ------------------------
    print("CALCULATE GPD DESCRIPTORS ...")
    #methodgpd = 0 #['JcJLdLLa_reduced', 'JLd_all']
    #pca_on = False #True
    #pca_model = '/home/althausc/master_thesis_impl/posedescriptors/out/08/27_13-49-24/modelpca64.pkl'
    logfile = os.path.join(logpath, '4-gpd.txt')
    
    if pca_on:
        if pca_model is None:
            raise ValueError("Please specify a pca model file for feature reduction.")
        os.system("python3.6 /home/althausc/master_thesis_impl/scripts/pose_descriptors/geometric_pose_descriptor.py -inputFile {} -mode {} -pcamodel {} &> {}"\
                                                                                                                .format(annpath, methodgpd, pca_model, logfile))
    else:
        os.system("python3.6 /home/althausc/master_thesis_impl/scripts/pose_descriptors/geometric_pose_descriptor.py -inputFile {} -mode {} &> {}"\
                                                                                                            .format(annpath, methodgpd, logfile))

    print("CALCULATE GPD DESCRIPTORS DONE.")
    out_dir = '/home/althausc/master_thesis_impl/posedescriptors/out/09'
    outrun_dir = latestdir(out_dir)

    gpdfile = filewithname(outrun_dir, 'geometric_pose_descriptor')
    return gpdfile

def search(gpdfile, method_search, gpdtype, tresh=None):
    # -------------------------- ELASTIC SEARCH -----------------------------
    print("SEARCH FOR GPD IN DATABASE:")

    inputfile = gpdfile
    logfile = os.path.join(logpath, '4-search.txt')
    print("GPD file: ",inputfile)
    _METHODS_SEARCH = ['COSSIM', 'DISTSUM']
    _GPD_TYPES = ['JcJLdLLa_reduced', 'JLd_all']

    assert method_search in _METHODS_SEARCH
    assert gpdtype in range(_GPD_TYPES)

    method_search = 'COSSIM' #['COSSIM', 'DISTSUM'] #testing
    gpdtype = _GPD_TYPES[gpdtype]

    #evaltresh_on = True

    #Querying on the database images
    if method_search == 'COSSIM':
        #Not implemented so far
        #res = input("Do you want to evaluate the treshold on the gpu clustering first? [yes/no]")
        #if res == 'yes':
        #    os.system("python3.6 /home/althausc/master_thesis_impl/retrieval/elastic_search_init.py -file {} -method_search {} -evaltresh".format(inputfile, method_search))
        if tresh is None:
            tresh = float(input("Please specify a similarity treshold for cossim result list: "))
        print("python3.6 /home/althausc/master_thesis_impl/retrieval/elastic_search_init.py -file {} -search -method_search {} \
                                                                                                -gpd_type {} -tresh {} &> {}"\
                                                                                                .format(inputfile, method_search, gpdtype, tresh, logfile))
        
        os.system("python3.6 /home/althausc/master_thesis_impl/retrieval/elastic_search_init.py -file {} -search -method_search {} \
                                                                                                -gpd_type {} -tresh {} &> {}"\
                                                                                                .format(inputfile, method_search, gpdtype, tresh, logfile))
    else:
        os.system("python3.6 /home/althausc/master_thesis_impl/retrieval/elastic_search_init.py -file {} -search --method_search {} \
                                                                                                -gpd_type {} &> {}"\
                                                                                                .format(inputfile, method_search, gpdtype, logfile))
    print('\n\n')

    outrun_dir = latestdir('/home/althausc/master_thesis_impl/retrieval/out/humanposes')
    print(outrun_dir)
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

def drawborder(imgpath):
    im = cv2.imread(imgpath)
    row, col = im.shape[:2]
    bottom = im[row-2:row, 0:col]

    bordersize = 7
    border = cv2.copyMakeBorder(
        im,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 0, 0]
    )

    #image_name = os.path.splitext(os.path.basename(imgpath))[0]
    #image_name = "%s_borders.jpg"%image_name
    
    #cv2.imwrite(os.path.join(os.path.dirname(imgpath), image_name), border)
    img = cv2.cvtColor(border, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)

    return img_pil

if __name__=="__main__":
   #test()
   drawborder('/home/althausc/master_thesis_impl/scripts/branchkpts/input_img.jpg')
   #print(getImgs('/home/althausc/master_thesis_impl/retrieval/out/09/02_13-16-41/result-ranking.json'))