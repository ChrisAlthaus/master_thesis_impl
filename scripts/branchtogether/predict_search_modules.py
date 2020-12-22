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

bgraphs_dir = '/home/althausc/master_thesis_impl/scripts'
import sys
sys.path.insert(0,bgraphs_dir)
import branchkpts.predict_search_modules as kptm
import branchgraphs.predict_search_modules as graphm

from multiprocessing import Process, Queue

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

def predict_kpts_and_scenegraph(imagepath):
    q = Queue()
    processes = [ Process(target = kptm.predict, args=(imagepath, q)),
                  Process(target = graphm.predict, args=(imagepath, q))]

    for p in processes:
        p.start()

    rets = []
    for p in processes:
        ret = q.get()
        rets.append(ret)

    for p in processes:
        p.join()

    rets_dict = {}
    for r in rets:
        rets_dict.update(r)
    print(rets_dict)
    return rets_dict

def transform_into_gpd_and_g2vfeature(kpt_annpath, g_annpath, methodgpd='JcJLdLLa_reduced', flipgpd=False, relsasnodes=True):
    pca_on = False #True
    pca_model = '/home/althausc/master_thesis_impl/posedescriptors/out/08/27_13-49-24/modelpca64.pkl' 
    
    q = Queue()
    processes = [ Process(target = kptm.transform_to_gpd, args=(kpt_annpath, methodgpd, pca_on, pca_model, flipgpd, q)),
                  Process(target = graphm.transform_into_g2vformat, args=(g_annpath, relsasnodes, q))]
    for p in processes:
        p.start()

    rets = []
    for p in processes:
        ret = q.get()
        rets.append(ret)

    for p in processes:
        p.join()

    rets_dict = {}
    for r in rets:
        rets_dict.update(r)
    print(rets_dict)
    return rets_dict['gpdfile'], rets_dict['graphfile']

def search(gpdfile, graphfile, gpdmsearch='L2', gpdrankingtype='max', gpdtype='JcJLdLLa_reduced', percperson=True):
    rw = False #True #not used
    rm = 'jaccard' #'euclid' #not used

    q = Queue()
    processes = [ Process(target = kptm.search, args=(gpdfile, gpdmsearch, gpdrankingtype, gpdtype, percperson, q)),
                  Process(target = graphm.search, args=(graphfile, rw, rm, q))]
    for p in processes:
        p.start()

    rets = []
    for p in processes:
        ret = q.get()
        rets.append(ret)

    for p in processes:
        p.join()

    rets_dict = {}
    for r in rets:
        rets_dict.update(r)
    print(rets_dict)
    return rets_dict['gpdrankingfile'], rets_dict['graphrankingfile']


def merge_retrievalresults(topkkpt_file, topngraph_file, topk=10, rankingmode='everynthweighted', weight_branches = 0.5):
    # Format of input files: 
    # {
    #"imagedir": "/home/althausc/nfs/data/coco_17_medium/train2017_styletransfer",
    #"0": {
    #    "filename": "000000359029_100645.jpg",
    #    "relscore": 1.0
    #},
    #"1": {
    #    "filename": "000000416510_062615.jpg",
    #    "relscore": 1.0 } ... 
    # }
    print("Reading keypoints from file: ",topkkpt_file)
    with open (topkkpt_file, "r") as f:
        kpt_data = json.load(f)

    kpt_imagedir = kpt_data['imagedir']
    del kpt_data['imagedir']

    print("Reading graphs from file: ",topngraph_file)
    with open (topngraph_file, "r") as f:
        graph_data = json.load(f)

    graph_imagedir = graph_data['imagedir']
    del graph_data['imagedir']

    #assert kpt_imagedir == graph_imagedir, 'Keypoint imagedir {} not equal to scene graph imagedir {}.'\
    #                                        .format(kpt_imagedir, graph_imagedir)

    kpt_data = sorted(list(kpt_data.items()), key=lambda x: x[0])
    graph_data = sorted(list(graph_data.items()), key=lambda x: x[0])

    mergedtopk = {'imagedir': [kpt_imagedir, graph_imagedir]}
    c_graphs = 0
    c_kpts = 0
   
    kpos = 0
    #First get all intersections
    for pos1,item1 in kpt_data:
        for pos2,item2 in graph_data:
            if os.path.basename(item1['filename']) == os.path.basename(item2['filename']):
                mergedtopk[str(kpos)] = {'filename': item1['filename'], 'krelscore': item1['relscore'], 'grelscore': item2['relscore'],
                                         'relscore': (item1['relscore'] + item2['relscore'])/2, 'posmerged': '{}{}'.format(pos1,pos2)}
                del graph_data[graph_data.index((pos2,item2))]
                del kpt_data[kpt_data.index((pos1,item1))]
                kpos = kpos + 1
    print("Number of intersections = {}".format(len(mergedtopk)-1))

    #Add label for seeing source of ranked items
    for _,item in graph_data:
        item.update({'type': 'graph'})
    for _,item in kpt_data:
        item.update({'type': 'kpts'})

    #Modes for the remaining topk image lists
    #   1. 'everynthweighted': take every nth image from either keypoints or graphs,
    #                          dependent on the weighting parameter
    #   2. 'highestscore': merge result & process in descreasing score order
    if rankingmode == 'everynthweighted':
        if weight_branches == 0.0:
            for k in range(0, topk):
                mergedtopk[str(kpos+k)] = kpt_data[k][1]

        elif weight_branches == 1.0:
            for k in range(0, topk):
                mergedtopk[str(kpos+k)] = graph_data[k][1]

        elif weight_branches<=0.5:
            everynthgraph = int(1/weight_branches)
            for k in range(0, topk):
                if (k+1)%everynthgraph == 0 and c_graphs<len(graph_data):
                    mergedtopk[str(k+kpos)] = graph_data[c_graphs][1]
                    c_graphs= c_graphs + 1
                else:
                    if c_kpts<len(kpt_data):
                        mergedtopk[str(k+kpos)] = kpt_data[c_kpts][1]
                        c_kpts = c_kpts + 1              
        else:
            everynthkpts = int(1/(1-weight_branches))
            for k in range(0, topk):
                if (k+1)%everynthkpts == 0 and c_kpts<len(kpt_data):
                    mergedtopk[str(k+kpos)] = kpt_data[c_kpts][1]
                    c_kpts = c_kpts + 1
                else:
                    if c_graphs<len(graph_data):
                        mergedtopk[str(k+kpos)] = graph_data[c_graphs][1]
                        c_graphs= c_graphs + 1

    elif rankingmode == 'highestscore':
        merged = kpt_data + graph_data
        merged.sort(key=lambda x: x[1]['relscore'], reverse=True)

        for _, item in merged:
             mergedtopk[str(kpos)] = item
             kpos = kpos + 1

    #Save merged results list to file
    savedir = '/home/althausc/master_thesis_impl/retrieval/out/bothtogether'
    outputdir = os.path.join(savedir, datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
    os.makedirs(outputdir)

    topkfile = os.path.join(outputdir, 'topkmerged.json')
    with open(topkfile, 'w') as f:
        json.dump(mergedtopk, f, indent=4, separators=(',', ': '))

    return topkfile

def getImgs(topkresults, drawgraphs=None, drawkpts=None):
    #topkresults format [(filepath1, score1), ... ]
    print("Reading from file: ",topkresults)
    with open (topkresults, "r") as f:
        json_data = json.load(f)

    imagedir = json_data['imagedir']
    del json_data['imagedir']

    topkdata = sorted(json_data.items(), key= lambda x: int(x[0])) 

    drawkptsdir = '/home/althausc/master_thesis_impl/detectron2/out/art_predictions/train/12-14_18-27-33/.visimages'
    drawgraphdir = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/out/predictions/graphs/12-10_17-57-13/.visimages'

    def getimg(imgpath):
        print("getting image: ",imgpath)
        try:
            img = Image.open(imgpath)
        except Exception as e: #Guard against too large images
            print(e)
            img = Image.new('RGB', (600, 400), (0,0,0))
        return img

    imgs = []
    scores = []
    for item in topkdata:
        print(item)
        if item[1]['type'] == 'kpts':
        #os.path.isfile(os.path.join(imagedir[0], item[1]['filename'])):
            if drawkpts:
                basename, suffix = os.path.splitext(item[1]['filename'])
                kfilename = '{}_overlay{}'.format(basename, suffix) 
                imgs.append(getimg(os.path.join(drawkptsdir, kfilename)))
                print("add: ",os.path.join(drawkptsdir, kfilename))
            else:
                imgs.append(getimg(os.path.join(imagedir[0], item[1]['filename'])))
                print("add: ",os.path.join(imagedir[0], item[1]['filename']))
        elif item[1]['type'] == 'graph':
            if drawgraphs:
                basename, suffix = os.path.splitext(item[1]['filename'])
                gfilename = '{}_1scenegraph{}'.format(basename, suffix)
                imgs.append(getimg(os.path.join(drawgraphdir, gfilename)))
                print("add: ",os.path.join(drawgraphdir, gfilename))
            else:
                im = getimg(os.path.join(imagedir[0], item[1]['filename']))
                print("add: ",os.path.join(imagedir[0], item[1]['filename']))
        else:
            raise ValueError()
        scores.append(item[1]['relscore'])

    return imgs, scores


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

    
    
 




