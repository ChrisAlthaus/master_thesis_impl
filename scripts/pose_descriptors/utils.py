import numpy as np
from sklearn.decomposition import PCA
import json
import warnings
import matplotlib.pyplot as plt
import os
import datetime
from collections import defaultdict
np.seterr(all = "raise")

import sys
sys.path.insert(0, '/home/althausc/master_thesis_impl/scripts/utils') 
from statsfunctions import getwhiskersvalues

def normalizevec(featurevector, rangemin=0, rangemax=1, mask=False):
    if mask:
        #Filter out unvalid features (=-1) for normalization
        #Then combine normalization and unvalid features in same ordering
        maskvalid = []
        for n in featurevector:
            if n == -1:
                maskvalid.append(False)
            else:
                maskvalid.append(True)
        subvec = [n for n,l in zip(featurevector, maskvalid) if l is True]
        if len(subvec)>0:
            if max(subvec) != min(subvec):
                normsubvec = [ (x-min(subvec)) * (rangemax - rangemin)/(max(subvec) - min(subvec)) + rangemin for x in subvec]
            else:
                #relation description not possible when same values or just on entry
                print("Info: relation description not possible")
                normsubvec = [-1 for _ in subvec]

        normvec = []
        c_valid = 0
        for l in maskvalid:
            if l is True:
                normvec.append(normsubvec[c_valid])
                c_valid += 1
            else:
                normvec.append(-1)
        return normvec
    else:
        #Normalize aka rescale input vector to new range
        return [ (x-min(featurevector)) * (rangemax - rangemin)/(max(featurevector) - min(featurevector)) + rangemin for x in featurevector]

def applyPCA(json_data, dim=None, pca=None):
    gpds = [item['gpd'] for item in json_data]
    if pca is None:
        pca = PCA(n_components=dim)
        pca_result = pca.fit_transform(gpds)
    else:
        pca_result = pca.transform(gpds)
    for i,item in enumerate(json_data):
        item['gpd'] = list(pca_result[i])
    
    return pca

def dict_to_item_list(d, level=1):
        #Ressolves double list items k:[[a],[b]] -> [k:[a]],[k:[b]] 
        items = []
        for k,item in d.items():
            if level==1:
                if isinstance(item[0], list):
                    for e in item:
                        items.append([k,e])
                else:
                    items.append([k,item])
            if level==0:
                if isinstance(item, list):
                    for e in item:
                        items.append([k,e])
                else:
                    items.append([k,item])
        return items

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]
    div = det(xdiff, ydiff)
    if div == 0:
        print('Info: lines do not intersect')
        return -1
    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return [x, y]

_COUNTER = 0
def plotsave_linesintersect(l1, l2, l3, l4, points):
        plt.plot([l1[0], l1[2]], [l1[1], l1[3]], 'g-', lw=1)
        plt.plot([l2[0], l2[2]], [l2[1], l2[3]], 'g-', lw=1)

        plt.plot([l3[0], l3[2]], [l3[1], l3[3]], 'r-', lw=1)
        plt.plot([l4[0], l4[2]], [l4[1], l4[3]], 'r-', lw=1)
        for p in points:
            plt.plot(p[0],p[1], marker='o', markersize=3, color="blue")
        global _COUNTER
        plt.savefig("/home/althausc/master_thesis_impl/posedescriptors/out/debugging/error_calcangle%d.jpg"%_COUNTER)
        _COUNTER = _COUNTER + 1
        plt.clf()

def angle(l1,l2):
    [p11,p12] = list(l1.coords)
    [p21,p22] = list(l2.coords)
    #print("coords1: ", list(l1.coords))
    #print("coords2: ", list(l2.coords))
    intersect = None
    #When line is a point (due to prediction overlay kpts) return 0
    if (p11[0] == p12[0] and p11[1] == p12[1]) or (p21[0] == p22[0] and p21[1] == p22[1]):
        return 0
    #Find intersection point for error prone angle computation
    if p11==p21:
        j1 = np.subtract(p12,p11)
        j2 = np.subtract(p22,p21)
    elif p11==p22:
        j1 = np.subtract(p12,p11)
        j2 = np.subtract(p21,p22)
    elif p12==p21:
        j1 = np.subtract(p11,p12)
        j2 = np.subtract(p22,p21)
    elif p12==p22:
        j1 = np.subtract(p11,p12)
        j2 = np.subtract(p21,p22)
    else:
        #Rare call to this computation
        #Don't using shapely cause no extended lines
        intersect = line_intersection(l1.coords, l2.coords)
        if intersect == -1: #parallel lines
            return 0.0
        #print("1:",p12, intersect)
        #print("2:",p22, intersect)

        j1 = np.subtract(p12,intersect)
        j2 = np.subtract(p22,intersect)

        #If intersection is the endpoint, take the startpoint
        if np.linalg.norm(j1) < 10e-4:
            j1 = np.subtract(p11,intersect)
            print("Info: intersection is exactly on the line endpoint")
            plotsave_linesintersect([0,0,j1[0],j1[1]], [0,0,j2[0],j2[1]], [p11[0],p11[1],p12[0],p12[1]], [p21[0],p21[1],p22[0],p22[1]], [intersect])
            assert np.linalg.norm(j1) > 10e-4
        if np.linalg.norm(j2) < 10e-4:
            j2 = np.subtract(p21,intersect)
            print("Info: intersection is exactly on the line endpoint")
            plotsave_linesintersect([0,0,j1[0],j1[1]], [0,0,j2[0],j2[1]], [p11[0],p11[1],p12[0],p12[1]], [p21[0],p21[1],p22[0],p22[1]], [intersect])
            assert np.linalg.norm(j2) > 10e-4
    
    #print(l1,list(l1.coords),l2,list(l2.coords))
    #print(j1,j2)
    try:
        j1_norm = j1/np.linalg.norm(j1)
        j2_norm = j2/np.linalg.norm(j2)
    except:
        plotsave_linesintersect([0,0,j1[0],j1[1]], [0,0,j2[0],j2[1]], [p11[0],p11[1],p12[0],p12[1]], [p21[0],p21[1],p22[0],p22[1]], [intersect])
        raise ValueError()
    #python3.6 /home/althausc/master_thesis_impl/scripts/pose_descriptors/geometric_pose_descriptor.py -inputFile /home/althausc/master_thesis_impl/detectron2/out/art_predictions/train/11-24_15-33-23/maskrcnn_predictions.json -mode JcJLdLLa_reduced -target insert

    return np.arccos(np.clip(np.dot(j1_norm, j2_norm), -1.0, 1.0))

def getimgfrequencies(descriptors):
    c_imgid_persons = defaultdict(int)
    for d in descriptors:
        imageid = d["image_id"]
        c_imgid_persons[imageid] =  c_imgid_persons[imageid] + 1
    return c_imgid_persons

def replace_unvalidentries(inputdescriptor, replacefeature):
    #Function replaces the missing pose descriptor values (-1's) with the values of the input feature
    for l in range(len(inputdescriptor)):
        if inputdescriptor[l] == -1:
            inputdescriptor[l] = replacefeature[l]

def get_neutral_pose_feature(mode='JcJLdLLa_reduced'):
    if mode == 'JcJLdLLa_reduced':
        ngpdpath = '/home/althausc/master_thesis_impl/posedescriptors/out/query/12-16_15-46-17/geometric_pose_descriptor_c_1_mJcJLdLLa_reduced_t0.05_f1_mkpt7n1.json'
    elif mode == 'JLd_all_direct':
        ngpdpath = '/home/althausc/master_thesis_impl/posedescriptors/out/query/12-16_15-46-17/geometric_pose_descriptor_c_1_mJcJLdLLa_reduced_t0.05_f1_mkpt7n1.json'
    elif mode == 'JJo_reduced':
        ngpdpath = '/home/althausc/master_thesis_impl/posedescriptors/out/query/12-16_15-46-17/geometric_pose_descriptor_c_1_mJcJLdLLa_reduced_t0.05_f1_mkpt7n1.json'
    else:
        raise ValueError()
     
    with open (ngpdpath, "r") as f:
        neutralgpd = json.load(f)[0]
    return neutralgpd['gpd']

def create_reference_feature(descriptors, mode='JcJLdLLa_reduced'):
    output_dir = os.path.join('/home/althausc/master_thesis_impl/posedescriptors/out/eval', datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
    os.makedirs(output_dir)

    refposdict = {}
    for k in range(len(descriptors[0]['gpd'])):
        refposdict[k] = []
    
    for desc in descriptors:
        for k,val in enumerate(desc['gpd']):
            if val != -1:
                refposdict[k].append(val)
    
    refposlists = sorted(refposdict.items(), key= lambda x: int(x[0])) 

    statsperindex = {}
    for k,indexlist in refposlists:
        statsperindex[k] = getwhiskersvalues(indexlist, mode='dict')
   
    with open(os.path.join(output_dir, 'features-per-index-statistics.json'), 'w') as f:
        print("Writing to folder: ",output_dir)
        json.dump(statsperindex, f, indent=1)

def get_reference_feature():
    featuresstats_path = '/home/althausc/master_thesis_impl/posedescriptors/out/eval/12-17_09-28-55-ref-feature/features-per-index-statistics.json'

    print("Reading from file: ",featuresstats_path)
    with open (featuresstats_path, "r") as f:
        featurestats = json.load(f)
    featurestats = sorted(featurestats.items(), key= lambda x: int(x[0])) 

    referencefeature = []
    for k,stats in featurestats:
        referencefeature.append(stats['median'])
    
    print("Reference feature: ", referencefeature)
    return referencefeature

def calc_reference_feature_total_median():
    import json
    file = '/home/althausc/master_thesis_impl/posedescriptors/out/eval/12-17_09-28-55-ref-feature/features-per-index-statistics.json'
    with open (file, "r") as f:
        json_data = json.load(f)
    medians = []
    for n, stats in json_data.items():
        medians.append(stats['median'])
    median = sum(medians)/len(medians)
    print("Median over all sub-medians: ",median)

def calc_neutralpose_median():
    if mode == 'JcJLdLLa_reduced':
        file = '/home/althausc/master_thesis_impl/posedescriptors/out/query/12-16_15-46-17/geometric_pose_descriptor_c_1_mJcJLdLLa_reduced_t0.05_f1_mkpt7n1.json'
    elif mode == 'JLd_all_direct':
        file = '/home/althausc/master_thesis_impl/posedescriptors/out/query/12-16_15-46-17/geometric_pose_descriptor_c_1_mJcJLdLLa_reduced_t0.05_f1_mkpt7n1.json'
    elif mode == 'JJo_reduced':
        file = '/home/althausc/master_thesis_impl/posedescriptors/out/query/12-16_15-46-17/geometric_pose_descriptor_c_1_mJcJLdLLa_reduced_t0.05_f1_mkpt7n1.json'
    else:
        raise ValueError()

    with open (file, "r") as f:
        json_data = json.load(f)
    neutralgpd = json_data[0]['gpd']
    print("Median of neutral pose feature: ", np.mean(neutralgpd))
    


   