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
import random
import shutil
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets


"""parser = argparse.ArgumentParser()
parser.add_argument('-inputImg',required=True,
                    help='Image for which to infer best k matching images.')
args = parser.parse_args()

#Folder with input image only
img_dir = os.path.dirname(args.inputImg)"""

logpath = '/home/althausc/master_thesis_impl/results/logs/jupyter-notebook/graphbranch'
if not os.path.exists(logpath):
    os.makedirs(logpath)
    print("Successfully created output directory: ", logpath)


def latestdir(dir):
    diritems = [os.path.join(dir, d) for d in os.listdir(dir)]
    all_subdirs = [d for d in diritems if os.path.isdir(d)]
    return max(all_subdirs, key=os.path.getmtime)

def filewithname(dir, searchstr):
    for item in os.listdir(dir):
        if os.path.isfile(os.path.join(dir,item)) and searchstr in item:
            return os.path.join(dir,item)
    return None

def predict(imagepath, queue=None):
    #Create a tmp image dir
    img_dir = os.path.join('/home/althausc/master_thesis_impl/scripts/branchgraphs/.images/singledirs', datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    else:
        raise ValueError("Output directory %s already exists."%img_dir)
    print(imagepath)
    shutil.copyfile(imagepath, os.path.join(img_dir, os.path.basename(imagepath)))

    # ----------------- SCENE GRAPH PREDICTION ---------------------
    print("SCENE GRAPH PREDICTION ...")
    gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun_G1d4-2.sh'
    #Note: model directory should contain a file 'last_checkpoint' with path to the used checkpoint
    model_dir = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/sgdet_training/12-02_09-23-52-dev3'
                #'/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/others/causal_motif_sgdet' 
                
    out_dir = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/out/predictions/single'
    logfile = os.path.join(logpath, '1-scenegraph.txt')

    _EFFECT_TYPES = ['none', 'TDE', 'NIE', 'TE']
    _FUSION_TYPES = ['sum', 'gate']
    _CONTEXTLAYER_TYPES = ['motifs', 'vctree', 'vtranse']

    effect_type = _EFFECT_TYPES[1]
    fusion_type = _FUSION_TYPES[0]
    contextlayer_type = _CONTEXTLAYER_TYPES[0] 

    topkboxes = -1 #10 #-1
    topkrels = 75 #20 #75
    treshboxes = 0.15
    treshrels = 0.15

    #print("Logfile: ", logfile)
    masterport = random.randint(10020, 10100)

    os.chdir('/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch')
    #Note: MODEL.PRETRAINED_DETECTOR_CKPT same functionality as OUTPUT_DIR (but OUTPUT_DIR used for model loading)
    #    
    cmd = ("{} python3.6 -m torch.distributed.launch" +\
                    "\t --master_port {}" +\
                    "\t --nproc_per_node=1 /home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/tools/relation_test_net.py" +\
                    "\t --config-file \"/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/configs/e2e_relation_X_101_32_8_FPN_1x.yaml\" " +\
                    "\t MODEL.ROI_RELATION_HEAD.USE_GT_BOX False" +\
                    "\t MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False" +\
                    "\t MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor" +\
                    "\t MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE {}" +\
                    "\t MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE {}" +\
                    "\t MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER {}" +\
                    "\t TEST.IMS_PER_BATCH 1" +\
                    "\t DTYPE \"float16\"" +\
                    "\t GLOVE_DIR /home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/sgdet_training/glove" +\
                    "\t MODEL.PRETRAINED_DETECTOR_CKPT {}" +\
                    "\t OUTPUT_DIR {}" +\
                    "\t TEST.CUSTUM_EVAL True" +\
                    "\t TEST.CUSTUM_PATH {}" +\
                    "\t TEST.POSTPROCESSING.TOPKBOXES {}" +\
                    "\t TEST.POSTPROCESSING.TOPKRELS {}" +\
                    "\t TEST.POSTPROCESSING.TRESHBOXES {}" +\
                    "\t TEST.POSTPROCESSING.TRESHRELS {}" +\
                    "\t DETECTED_SGG_DIR {} \t &> {}").format(gpu_cmd, masterport, effect_type, fusion_type, contextlayer_type, model_dir, model_dir, img_dir,
                                                              topkboxes, topkrels, treshboxes, treshrels, out_dir, logfile)
    print(cmd)               
    if os.system(cmd):
        raise RuntimeError('Scene graph prediction failed.')

    print("SCENE GRAPH PREDICTION DONE.\n")
    predicdir = latestdir(out_dir)

    print("VISUALIZE SCENEGRAPH ...")
    #Whether to show only valid labels or all labels from prediction
    logfile = os.path.join(logpath, '2-visualize.txt')

    cmd = "python3.6 /home/althausc/master_thesis_impl/scripts/scenegraph/visualizeimgs.py -predictdir {} &> {}"\
                                .format(predicdir, logfile)
    print(cmd)
    if os.system(cmd):
        raise RuntimeError('Scene graph visualization failed.')

    print("VISUALIZE SCENEGRAPH DONE.\n")
    outrun_dir = latestdir('/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/out/predictions/single')
                        #'/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/out/visualize'
    outrun_dir = os.path.join(outrun_dir, '.visimages')
    files = [os.path.join(outrun_dir, f) for f in os.listdir(outrun_dir) if os.path.isfile(os.path.join(outrun_dir, f))]
    imgpath, ann = sorted(files, key=lambda x: os.path.splitext(os.path.basename(x))[0])
    #print(sorted(files, key=lambda x: os.path.splitext(os.path.basename(x))[0]))
    annstr = ''
    with open(ann, 'r') as f:
        annstr = f.read()

    if queue:
        queue.put({'scenegraph':[predicdir, imgpath, annstr]})
        return
    return predicdir, imgpath, annstr

def transform_into_g2vformat(anndir, relasnodes=True, queue=None):
    # ----------------- TRANSFORM PREDICTIONS INTO GRAPH2VEC FORMAT ---------------
    print("TRANSFORM PREDICTIONS INTO GRAPH2VEC FORMAT ...")

    pred_imginfo = os.path.join(anndir, 'custom_data_info.json')
    pred_file = os.path.join(anndir, 'custom_prediction.json')
    logfile = os.path.join(logpath, '3-transform.txt')
    
    cmd = "python3.6 /home/althausc/master_thesis_impl/scripts/graph_descriptors/graphdescriptors.py \
                    -file {} \
                    -imginfo {} \
                    {} &> {}".format(pred_file, pred_imginfo, '-relsasnodes' if relasnodes else ' ', logfile)
    print(cmd)
    if os.system(cmd):
        raise RuntimeError('Transform predictions failed.')

    graphfile = os.path.join(anndir, '.descriptors', 'graphdescriptors.json')
    print("TRANSFORM PREDICTIONS INTO GRAPH2VEC FORMAT DONE.")
    print("Graphfile: ",graphfile)

    if queue:
        queue.put({'graphfile': graphfile})
        return
    return graphfile

def search(graphfile, reweight=False, r_mode='jaccard', steps=3000, queue=None):
    # ----------------- GRAPH2VEC PREDICTION & RETRIEVAL ---------------------
    print("GRAPH2VEC PREDICTION & RETRIEVAL ...")
    g2v_model = '/home/althausc/master_thesis_impl/graph2vec/models/12-16_11-47-55/g2vmodelc65839d128e40'
                #'/home/althausc/master_thesis_impl/graph2vec/models/12-14_17-51-56/g2vmodelc65839d128e40'
                #'/home/althausc/master_thesis_impl/graph2vec/models/12-12_10-34-58-kaggle-from-here-on/g2vmodelc65839d128e40'
                #'/home/althausc/master_thesis_impl/graph2vec/models/12-11_11-11-43/g2vmodelc9211d128e100'
                #'/home/althausc/master_thesis_impl/graph2vec/models/12-10_10-31-58/g2vmodelc9211d1024e200'

    graphdir = os.path.dirname(graphfile)
    labelvecpath = os.path.join(graphdir, 'labelvectors-topk.json')
    inputfile = graphfile   #'/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/out/topk/single/09-25_15-23-04/graphs-topk.json'
                            #'/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/out/topk/single/09-23_14-52-00/graphs-topk.json' #graphfile
    print('inputfile:', inputfile)
    topk = 100
    logfile = os.path.join(logpath, '4-retrieval.txt')

    wliters = 3
    min_featuredim = 100 #136 #-1 #136
    #steps = 2000
    print("wliters: ", wliters)

    if reweight:
        if os.system("python3.6 /home/althausc/master_thesis_impl/retrieval/graph_search.py \
                        --model {} --inputpath {} \
    				    --inference --wl-iterations {} --steps-infer {} --topk {} \
                        --reweight --reweightmode {} \
                        --labelvecpath {} &> {}".format(g2v_model, inputfile, wliters, steps, topk, r_mode, labelvecpath, logfile)):
            raise RuntimeError('Scene graph search failed.')            
    else:
        cmd = ("python3.6 /home/althausc/master_thesis_impl/retrieval/graph_search.py --model {} --inputpath {} " +\
    				 "--inference --min-featuredim {} --steps-infer {} --topk {} &> {}").format(g2v_model, inputfile, min_featuredim, steps, topk, logfile)
        print(cmd)
        if os.system(cmd):
            raise RuntimeError('Scene graph search failed.')         

    out_dir = '/home/althausc/master_thesis_impl/retrieval/out/scenegraphs/'
    outrun_dir = latestdir(out_dir)

    print("GRAPH2VEC PREDICTION & RETRIEVAL DONE.")
    filename = os.path.join(outrun_dir, "topkresults.json")
    with open(filename, 'r') as f:
        json_data = json.load(f)
        #print("Results: ", json_data)
    print("Graph search returned {} results.".format(len(json_data)-1))

    if queue:
        queue.put({'graphrankingfile': filename})
        return

    return filename

def getImgs(topkresults, drawgraphs = False):
    #topkresults format [(filepath1, score1), ... ]

    print("Reading from file: ",topkresults)
    with open (topkresults, "r") as f:
        json_data = json.load(f)

    imagedir = json_data['imagedir']
    del json_data['imagedir']

    if drawgraphs:
        graphvisdir = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/out/predictions/graphs/12-08_18-42-37/.visimages'
        imagedir = graphvisdir

    rankedlist = sorted(json_data.items(), key= lambda x: int(x[0]))
    imgs = []
    scores = []
    for item in rankedlist:
        if drawgraphs:
            basename, suffix = os.path.splitext(item[1]['filename'])
            gfilename = '{}_1scenegraph{}'.format(basename, suffix) 
            imgs.append(Image.open(os.path.join(imagedir,gfilename)))
        else:
            basewidth = 512
            img = Image.open(os.path.join(imagedir, item[1]['filename'])).convert('RGB')
            wpercent = (basewidth/float(img.size[0]))
            hsize = int((float(img.size[1])*float(wpercent)))
            img = img.resize((basewidth,hsize), Image.ANTIALIAS)
            imgs.append(img)
        scores.append(item[1]['relscore'])
    
    return imgs, scores, rankedlist


def getimg(imgpath):
    img = Image.open(imgpath).convert('RGB')
    return img


def generateRandomRankedlists(num, k, savedir):
    import sys
    sys.path.append('/home/althausc/master_thesis_impl/scripts/detectron2/utils')
    import utilspbn
    
    return utilspbn.randomrankings(num, k, savedir)


def getRankedFilesFromMetadata():
    imgdir = '/home/althausc/nfs/data/userstudy-tests/retrieval-previous/data/scenegraphs/metadata'
    mpaths = [os.path.join(imgdir, f) for f in os.listdir(imgdir)]
    mpaths.sort()
    print(mpaths)
    rankedlists = []

    for mpath in mpaths:
        with open(mpath) as json_file:
            mdata = json.load(json_file)
        rankedlists.append(mdata['resultpath'])
    return rankedlists

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

    img = cv2.cvtColor(border, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)

    return img_pil

def treshIndex(tresh, rankedlist):
    with open (rankedlist, "r") as f:
        data = json.load(f)
    
    del data['imagedir']

    k = 0
    for item in data.values():
        print(item)
        if item['relscore']>= tresh:
            k = k + 1
    return k

def cropImage(imagepath, p1, p2, resize=True):
    outfile = os.path.join('.images', os.path.splitext(imagepath)[0] + "_transformed.jpg")
    img = Image.open(imagepath)
    area = (p1[0], p1[1], p2[0], p2[1])
    cropped_img = img.crop(area)
    width, height = cropped_img.size
    if resize:
        maxwidth, maxheight = 512,512
       
        ratio = min(maxwidth/width, maxheight/height)
        newsize = np.asarray(cropped_img.size) * ratio
        newsize = tuple(newsize.astype(int))
        
        cropped_img = cropped_img.resize(newsize, Image.ANTIALIAS)
        cropped_img.save(outfile, "JPEG")
        print("Cropped & Resized image to file: ",outfile)
    else:
        cropped_img.save(outfile, "JPEG")
        print("Cropped image to file: ",outfile)

    return outfile

    
    
 




