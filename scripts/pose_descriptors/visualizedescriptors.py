import time
import argparse
import json
import logging
from detectron2.structures import Instances
from detectron2.structures import Boxes
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.builtin_meta import KEYPOINT_CONNECTION_RULES, COCO_PERSON_KEYPOINT_NAMES, COCO_PERSON_KEYPOINT_FLIP_MAP
import os
import cv2
import torch
import itertools
import random
from utils import dict_to_item_list
import math
import matplotlib.pyplot as plt

#Visualizes human pose keypoints given by input file.
#Treshold for score possible.
#Image folder corresponding to input file annotations needed.

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-predictionfile',required=True,
                        help='File with a list of dict items with fields imageid and keypoints.')  
    parser.add_argument('-gpdfile',required=True) 
    parser.add_argument('-imagespath',required=True)   
    parser.add_argument('-transformid',action="store_true", 
                        help='Wheather to split imageid to get image filepath (used for style transfered images.')   
    parser.add_argument('-drawkptsonly',action="store_true")  
    parser.add_argument('-vistresh',type=float, default=0.0)                
    args = parser.parse_args()

    preddata = None
    with open(args.predictionfile, "r") as f:
         preddata = json.load(f)   

    gpddata = None
    with open(args.gpdfile, "r") as f:
         gpddata = json.load(f)   
    
    print("Sample at position 0: ", gpddata[0])
    print("Feature dimension: ", len(gpddata[0]['gpd']))

    predgpd_map = {}
    for item in preddata:
        if item['image_id'] not in predgpd_map:
            predgpd_map[item['image_id']] = [item]
        else:
            predgpd_map[item['image_id']].append(item)
    for item in gpddata:
        if item['image_id'] not in predgpd_map:
            predgpd_map[item['image_id']] = [item]
        else:
            predgpd_map[item['image_id']].append(item)

    #print(predgpd_map)
    outputdir = os.path.join(os.path.dirname(args.gpdfile), '.visimages')
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
        print("Successfully created output directory: ", outputdir)
    
    MetadataCatalog.get("my_dataset_val").set(keypoint_names=COCO_PERSON_KEYPOINT_NAMES,
                                              keypoint_flip_map=COCO_PERSON_KEYPOINT_FLIP_MAP,
                                              keypoint_connection_rules=KEYPOINT_CONNECTION_RULES)
    
    print("Visualize the predictions onto the original image(s) ...")
    if args.drawkptsonly:
        visualize_keypoints(predgpd_map, args.imagespath, outputdir)
        return

    gpdfilename = os.path.basename(args.gpdfile)
    _MODES = ['JcJLdLLa_reduced', 'JLd_all_direct', 'JJo_reduced']
    if 'JcJLdLLa_reduced' in gpdfilename:
        print("Detected descriptor type: JcJLdLLa_reduced")
        visualizeJcJLdLLa(predgpd_map, args.imagespath, outputdir)
    elif 'JLd_all_direct' in gpdfilename:
        print("Detected descriptor type: JLd_all_direct")
        visualizeJLdall(predgpd_map, args.imagespath, outputdir)
    elif 'JJo_reduced' in gpdfilename:
        print("Detected descriptor type: JJo_reduced")
        visualizeJJo(predgpd_map, args.imagespath, outputdir)
    elif 'Jc_rel' in gpdfilename:
        print("Detected descriptor type: Jc_rel")
        visualizeJcrel(predgpd_map, args.imagespath, outputdir)

    print("Visualize done.")
    print("Wrote images to path: ",outputdir)


body_part_mapping = {
        0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear", 5: "left_shoulder", 6: "right_shoulder",
        7: "left_elbow", 8: "right_elbow", 9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
        13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"}

def visualize_keypoints(predgpds, imagedir, outputdir, vistresh=0.0, transformid=False):
    for imgid, group in predgpds.items():
        imgid = str(imgid)
        preds = [item for item in group if 'bbox' in item or 'keypoints' in item]
        
        imgname = imgid
        imgname_out = os.path.basename(imgname)
        if not os.path.isabs(imgname):
            img_path = os.path.join(imagedir, imgname)
        else:
            img_path = imgname

        v = Visualizer(cv2.imread(img_path)[:, :, ::-1],MetadataCatalog.get("my_dataset_val"), scale=1.2)
        print("draw kpts")
        out = drawkeypoints(preds, img_path, v)
        print("write")
        cv2.imwrite(os.path.join(outputdir, imgname_out), out.get_image()[:, :, ::-1])

def visualizeJLdall(predgpds, imagedir, outputdir, vistresh=0.0, transformid=False):  
    l_direct_adjacent = [(0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (6, 5), (6, 12), (5, 11), (12, 11)]
    line_mapping = l_direct_adjacent

    display_percent = 0.075
    for imgid, group in predgpds.items():
        imgid = str(imgid)
        preds = [item for item in group if 'bbox' in item or 'keypoints' in item]
        gpds = [item for item in group if 'gpd' in item]

        imgname = imgid
        imgname_out = os.path.basename(imgname)
        if not os.path.isabs(imgname):
            img_path = os.path.join(imagedir, imgname)
        else:
            img_path = imgname

        keypoints = []
        for pred in preds:
            kpts = list(zip(pred['keypoints'][::3], pred['keypoints'][1::3], pred['keypoints'][2::3]))
            keypoints.append(kpts)

        if len(keypoints) != len(gpds):
            print("No 1-to-1 assignments of predictions->gpds possible, because filtered predictions by gpd calculation.")
            continue

        jld_kpts = []
        for i in range(len(keypoints)):
            kpts = keypoints[i]
            gpd = gpds[i]['gpd']
            gpdindex = 0

            for k1,k2 in line_mapping:
                for j,joint in enumerate(kpts):
                    #if joint is the same as either start or end point of the line, continue
                    if j==k1 or j==k2: 
                        continue
                    #print('%s->(%s,%s) %f'%(body_part_mapping[j], body_part_mapping[k1], body_part_mapping[k2], gpd[gpdindex]))
                    if random.random() < display_percent:
                        jltuple = [kpts[j][:2], kpts[k1][:2], kpts[k2][:2], gpd[gpdindex]]
                        jld_kpts.append(jltuple)
                    gpdindex = gpdindex + 1

        
        v = Visualizer(cv2.imread(img_path)[:, :, ::-1],MetadataCatalog.get("my_dataset_val"), scale=1.2)
        drawkeypoints(preds, img_path, v)
        outjldist = v.draw_gpddescriptor_jldist(jld_kpts)
        cv2.imwrite(os.path.join(outputdir, imgname_out+'_jLdall_{}.jpg'.format(display_percent)), outjldist.get_image()[:, :, ::-1])
    


def visualizeJcJLdLLa(predgpds, imagedir, outputdir, vistresh=0.0, transformid=False):
    #Grouped imageid input: [{imageid1 : [{imageid1,...},...,{imageid1,...}], ... , {imageidn :[{imageidn,...},...,{imageidn,...}]}]
    #Dimensions: 18 distances
    kpt_line_mapping = {7:[(5,9),'left_arm'], 8:[(6,10),'right_arm'], 
                                3:[(5,0),'shoulder_head_left'], 4:[(6,0),'shoulder_head_right'],
                                6:[[(8,5),'shoulders_elbowr'], [(10,4),'endpoints_earhand_shoulder_r']], 
                                5:[[(6,7),'shoulders_elbowsl'], [(3,9),'endpoints_earhand_shoulder_l']], 
                                13:[[(14,15),'knees_foot_side'], [(11,15),'left_leg']],
                                14:[[(13,16),'knees_foot_side'], [(12,16),'right_leg']], 
                                10:[(5,9),'arms_left_side'], 9:[(6,10),'arms_right_side'],
                                0:[[(16,12),'headpos_side'], [(15,11),'headpos_side']],
                                11:[(15,9),'endpoints_foodhand_hip_l'], 12:[(10,16),'endpoints_foodhand_hip_r']} 

    #Dimensions: 12 angles
    line_line_mapping = {(10,9):[[(9,15),'rhand_lhandrfoot'],[(9,16),'rhand_lhandlfoot'],
                                        [(10,16),'lhand_rhandlfoot'],[(10,15),'lhand_rhandrfoot']],
                                (5,11):[(5,9),'hand_shoulder_hip_l'], (6,12):[(6,10),'hand_shoulder_hip_r'],
                                (6,8):[(5,7),'upper_arms'], (8,10):[(7,9),'lower_arms'],
                                (12,14):[(11,13),'upper_legs'], (14,16):[(13,15),'lower_legs'],
                                (0,5):[(3,5),'head_shoulder_l'], (0,6):[(4,6),'head_shoulder_r']}
    
    for imgid,group in predgpds.items():
        imgid = str(imgid)
        preds = [item for item in group if 'bbox' in item or 'keypoints' in item]
        gpds = [item for item in group if 'gpd' in item]

        imgname = imgid
        imgname_out = os.path.basename(imgname)
        if not os.path.isabs(imgname):
            img_path = os.path.join(imagedir, imgname)
        else:
            img_path = imgname

        keypoints = []
        for pred in preds:
            kpts = list(zip(pred['keypoints'][::3], pred['keypoints'][1::3], pred['keypoints'][2::3]))
            keypoints.append(kpts)

        jld_kpts = []
        lla_kpts = []
        jl_start = 34
        ll_start = 52
        
        if len(keypoints) != len(gpds):
            print("No 1-to-1 assignments of predictions->gpds possible, because filtered predictions by gpd calculation.")
            continue

        for i in range(len(keypoints)):
            kpts = keypoints[i]
            gpd = gpds[i]['gpd']
            k = 0
            for j,l in dict_to_item_list(kpt_line_mapping):
                if isinstance(l[0], list):
                    for ls in l:
                        jltuple = [kpts[j][:2], kpts[ls[0][0]][:2], kpts[ls[0][1]][:2], gpd[jl_start+k]]
                        jld_kpts.append(jltuple)
                        k += 1
                else:
                    jltuple = [kpts[j][:2], kpts[l[0][0]][:2], kpts[l[0][1]][:2], gpd[jl_start+k]]
                    jld_kpts.append(jltuple)
                    k += 1
            k = 0
            for j,l in dict_to_item_list(line_line_mapping):
                #print("%d: (%s,%s)->(%s,%s)"%(k,body_part_mapping[j[0]], body_part_mapping[j[1]], body_part_mapping[l[0][0]], body_part_mapping[l[0][1]]))
                lltuple = [kpts[j[0]][:2] ,kpts[j[1]][:2], kpts[l[0][0]][:2], kpts[l[0][1]][:2],  gpd[ll_start+k]]
                lla_kpts.append(lltuple)
                k += 1

        v = Visualizer(cv2.imread(img_path)[:, :, ::-1],MetadataCatalog.get("my_dataset_val"), scale=1.2)
        drawkeypoints(preds, img_path, v)
        outjldist = v.draw_gpddescriptor_jldist(jld_kpts)
        cv2.imwrite(os.path.join(outputdir, imgname_out+'_jldists.jpg'),outjldist.get_image()[:, :, ::-1])

        v = Visualizer(cv2.imread(img_path)[:, :, ::-1],MetadataCatalog.get("my_dataset_val"), scale=1.2)
        drawkeypoints(preds, img_path, v)
        outllangle = v.draw_gpddescriptor_llangle(lla_kpts)
        cv2.imwrite(os.path.join(outputdir, imgname_out+'_llangles.jpg'),outllangle.get_image()[:, :, ::-1])

def visualizeJJo(predgpds, imagedir, outputdir, vistresh=0.0, transformid=False):
    #Grouped imageid input: [{imageid1 : [{imageid1,...},...,{imageid1,...}], ... , {imageidn :[{imageidn,...},...,{imageidn,...}]}]
    _KPTS_LINES = [(0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15)]   
    _KPTS_LINES.extend([(6,5), (6,12), (5,11), (12,11)]) 

    _END_JOINTS_DEPTH_2 = [(0,4), (0,3), (6,10), (5,9), (12,16), (11,15)]
    custom_jj_mapping = [(9,15), (10,16), #endjoints
                             (0,6), (0,5)] #head-shoulders
    kpt_kpt_mapping = []
    kpt_kpt_mapping.extend(_KPTS_LINES)
    kpt_kpt_mapping.extend(_END_JOINTS_DEPTH_2)
    kpt_kpt_mapping.extend(custom_jj_mapping)

    for imgid,group in predgpds.items():
        imgid = str(imgid)
        preds = [item for item in group if 'bbox' in item or 'keypoints' in item]
        gpds = [item for item in group if 'gpd' in item]

        imgname = imgid
        imgname_out = os.path.basename(imgname)
        if not os.path.isabs(imgname):
            img_path = os.path.join(imagedir, imgname)
        else:
            img_path = imgname

        keypoints = []
        for pred in preds:
            kpts = list(zip(pred['keypoints'][::3], pred['keypoints'][1::3], pred['keypoints'][2::3]))
            keypoints.append(kpts)

        if len(keypoints) != len(gpds):
            print("No 1-to-1 assignments of predictions->gpds possible, because filtered predictions by gpd calculation.")
            continue

        kptsjj_os = []
        kptdists = [] #for scaling arrows according to limb length
        for i in range(len(keypoints)):
            kpts = keypoints[i]
            gpd = gpds[i]['gpd']
            #print("kpts: ",kpts)
            #print("gpd: ",gpd)

            k = 0
            for j1,j2 in kpt_kpt_mapping:
                kptstart = kpts[j1]
                orientation = gpd[k:k+2] 
                kptsjj_os.append((kptstart, orientation))
                kptdists.append(math.hypot(kpts[j2][0]- kpts[j1][0],  kpts[j2][1]- kpts[j1][1]))
                k += 2
        #print("kptsjj_os: ",kptsjj_os)
        #print("kptdists: ",kptdists)
        print(img_path)
        v = Visualizer(cv2.imread(img_path)[:, :, ::-1],MetadataCatalog.get("my_dataset_val"), scale=1.2)
        drawkeypoints(preds, img_path, v)
        outjldist = v.draw_gpddescriptor_jjo(kptsjj_os, kptdists)
        cv2.imwrite(os.path.join(outputdir, imgname_out+'_jjo.jpg'),outjldist.get_image()[:, :, ::-1])

def visualizeJcrel(predgpds, imagedir, outputdir, vistresh=0.0, transformid=False):
    keypoint_connections = MetadataCatalog.get("my_dataset_val").keypoint_connection_rules
    keypoint_names = MetadataCatalog.get("my_dataset_val").get("keypoint_names")
    kconnections_indices = [(keypoint_names.index(l1), keypoint_names.index(l2), color) for l1,l2,color in keypoint_connections]

    for imgid,group in predgpds.items():
        imgid = str(imgid)
        preds = [item for item in group if 'bbox' in item or 'keypoints' in item]
        gpds = [item for item in group if 'gpd' in item]

        imgname = imgid
        imgname_out = os.path.basename(imgname)
        if not os.path.isabs(imgname):
            img_path = os.path.join(imagedir, imgname)
        else:
            img_path = imgname

        keypoints = []
        for pred in preds:
            kpts = list(zip(pred['keypoints'][::3], pred['keypoints'][1::3], pred['keypoints'][2::3]))
            keypoints.append(kpts)

        if len(keypoints) != len(gpds):
            print("No 1-to-1 assignments of predictions->gpds possible, because filtered predictions by gpd calculation.")
            continue

        plt.axis('equal')
        plt.gca().invert_yaxis()
        x_max = 0

        for i in range(len(keypoints)):
            kpts = keypoints[i]
            gpd = gpds[i]['gpd']
            kpoints = [(xrel*100 +x_max, yrel*100) if xrel!=-1 and yrel!=-1 else (-1,-1) for xrel, yrel in zip(gpd[::2], gpd[1::2])]

            for k1,k2,color in kconnections_indices:
                x = kpoints[k1]
                y = kpoints[k2]
                if x[0]>x_max:
                    x_max = x[0]
                if x[1]>x_max:
                    x_max = x[1]

                if x[0]==-1 or x[1]==-1 or y[0]==-1 or y[1]==-1:
                    continue
                #print([int(x[0]), int(y[0])], [int(x[1]), int(y[1])], (color[0]/255, color[1]/255, color[2]/255))
                plt.plot([int(x[0]), int(y[0])], [int(x[1]), int(y[1])], color=(color[0]/255, color[1]/255, color[2]/255), lw=1.5)
                x_max = x_max + 4

        plt.tight_layout()
        plt.box(False)
        plt.axis('off')
        plt.savefig(os.path.join(outputdir, imgname_out+'_jcrel.jpg'))
        plt.clf()
        plt.cla()

def drawkeypoints(preds, img_path, visualizer, vistresh=0.0):
     img = cv2.imread(img_path, 0)
     height, width = img.shape[:2]

     instances = Instances((height, width))
     boxes = []
     scores = []
     classes = []
     masks = []
     keypoints = []

     #"image_id": 785050351, "category_id": 1, "score": 1.0, "keypoints"
     for pred in preds:
         classes.append(pred["category_id"])
         if 'score' in pred: #gt annotations don't have score entry
             scores.append(pred['score'])
         else:
             scores.append(1.0)
         kpts = list(zip(pred['keypoints'][::3], pred['keypoints'][1::3], pred['keypoints'][2::3]))
         keypoints.append(kpts)
 
     instances.scores = torch.Tensor(scores)
     instances.pred_classes = torch.Tensor(classes)
     instances.pred_keypoints = torch.Tensor(keypoints)
     print("test")
     
     return visualizer.draw_instance_predictions(instances, vistresh)

if __name__=="__main__":
    main()