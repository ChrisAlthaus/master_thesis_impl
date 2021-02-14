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


def merge_retrievalresults(topkkpt_file, topngraph_file, topk=10, weight_branches = 0.5):
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

    intersection = []
    #First get all intersections
    for pos1,item1 in list(kpt_data.items()):
        for pos2,item2 in list(graph_data.items()):
            if os.path.basename(item1['filename']) == os.path.basename(item2['filename']):
                intersection.append(item1['filename'])

    #Order the remaining images by their individual score, weighted by importance factor
    # weight_branches:  0-> only consider keypoint branch
    #                   1-> only consider graph branch

    scores_kpts = [item['relscore'] for pos,item in list(kpt_data.items())]
    scores_graphs = [item['relscore'] for pos,item in list(graph_data.items())]

    assert min(scores_kpts) == 0.5 and max(scores_kpts) == 1, "Scores not in normalized range [0.5, 1]."
    assert min(scores_graphs) == 0.5 and max(scores_graphs) == 1, "Scores not in normalized range [0.5, 1]."
    assert weight_branches>= 0 and weight_branches<=1

    additional = []
    for pos,item1 in list(kpt_data.items()):
        if item1['filename'] in intersection:
            continue
        else:
            item = copy.deepcopy(item1)
            item['relscore'] = item['relscore']*(1-weight_branches)
            additional.append([item['filename'], item['relscore']])

    for item2 in graph_data:
        if item2[0] in intersection:
            continue
        else:
            item = copy.deepcopy(item2)
            item['relscore'] = item['relscore']*weight_branches
            #item[1] = item[1]*weight_branches
            additional.append([item['filename'], item['relscore']])

    additional = sorted(additional, key=lambda x: x[1], reverse=True)

    merged_topk = intersection[:topk]
    merged_topk.extend(additional[:topk-len(merged_topk)])

    return imagedir,merged_topk

def getImgs(imagedir, topkresults):
    #topkresults format [(filepath1, score1), ... ]

    #print("Reading from file: ",topkresults)
    #with open (topkresults, "r") as f:
    #    topkdata = json.load(f)
    topkdata = topkresults

    imgs = []
    scores = []
    for item in topkdata:
        imgs.append(Image.open(os.path.join(imagedir,item[0])))
        scores.append(item[1])
    
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

    #image_name = os.path.splitext(os.path.basename(imgpath))[0]
    #image_name = "%s_borders.jpg"%image_name
    
    #cv2.imwrite(os.path.join(os.path.dirname(imgpath), image_name), border)
    img = cv2.cvtColor(border, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)

    return img_pil

def testnormalizations():

    a = [0.1, 0.85, 0.85, 0.85, 0.85, 0.85, 0.9, 0.9, 0.9, 0.95]
    ax = [0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.9, 0.9, 0.9, 0.95]
    b = [0.1, 0.1, 0.1, 0.1,0.1, 0.1, 0.1, 0.1, 0.1, 0.95]
    c = [0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.61]
    d = [0.5, 0.6, 0.7, 0.7, 0.75, 0.8, 0.8, 0.8, 0.8, 0.81]
    e = [0.59, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.61]

    for array in [a, ax, b, c, d, e]:
        result1 = []
        result2 = []
        result3 = []
        for i in array:
            score_minmax =  (i - np.min(array))/(np.max(array) - np.min(array))  
            result1.append(score_minmax) 

            score_sumnorm =  (i - np.min(array))/(np.sum(np.array(array) - np.min(array)))
            result2.append(score_sumnorm) 

            zscore = (i- np.mean(array))/np.std(array)
            result3.append(zscore)

        print(array)
        print(result1)
        print(result2)
        print(result3)
        print("------------------------")



if __name__ == "__main__":
    testnormalizations()

    