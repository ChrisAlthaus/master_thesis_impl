#Creates custom features from skeleton pose annotations based on geometric properties.
import os
import json 
import argparse
import numpy as np
import math
from shapely.geometry import LineString, Point, Polygon
import scipy
from sklearn.decomposition import PCA
import pickle
import collections
import datetime
import time
import logging
import itertools
import copy
import matplotlib.pyplot as plt
from utils import applyPCA, normalizevec, dict_to_item_list, getimgfrequencies, line_intersection, angle

#Transform the input pose predictions to descriptors based on multiple measures.
#Provided measures:
#   - joint coordinates
#   - joint-joint distances (subset possible)
#   - joint-joint orientations (subset possible)
#   - joint-line distances (subset possible)
#   - joint-plane distances
#   - line-line angles
#Additional: To get a wanted descriptor dimension PCA can be used either 
#with a pre-trained 

parser = argparse.ArgumentParser()
parser.add_argument('-inputFile',required=True,
                    help='File with keypoint annotations/ predictions.')
parser.add_argument("-gtAnn", action='store_true', help="If input file is a groundtruth file. Used for testing and debugging.")
parser.add_argument("-mode", type=str, help="Specify types of features which will be computed.")
parser.add_argument("-pca", type=int, help="Specify dimensions of pca vector.")
parser.add_argument("-pcamodel", type=str, help="Specify pca model file for prediction.")
parser.add_argument("-target", type=str, help="If purpose is for inserting or query db. Used for output folder selection.")
parser.add_argument("-flip", action='store_true', help="Weather to flip the input predictions. Only for query mode.")

args = parser.parse_args()

#test_entry = {'image_id': 23899057496, 'category_id': 1, 'bbox': [211.99081420898438, 139.43743896484375, 425.96087646484375, 355.24871826171875], 'keypoints': [334.2212219238281, 201.67015075683594, 1.079627275466919, 331.54656982421875, 189.38385009765625, 1.7378227710723877, 312.2892761230469, 192.58897399902344, 1.028214931488037, 334.7561340332031, 202.20433044433594, 0.08344336599111557, 269.4952697753906, 213.9564208984375, 0.38487914204597473, 346.5245056152344, 262.033203125, 0.13131119310855865, 288.2176513671875, 285.5373840332031, 0.10808556526899338, 425.1584777832031, 354.4474182128906, 0.020250316709280014, 383.434326171875, 328.8064880371094, 0.012223891913890839, 276.44927978515625, 354.4474182128906, 0.01989334262907505, 425.1584777832031, 354.4474182128906, 0.020259613171219826, 425.1584777832031, 354.4474182128906, 0.02405051700770855, 403.761474609375, 354.4474182128906, 0.02277219668030739, 425.1584777832031, 354.4474182128906, 0.03073735162615776, 425.1584777832031, 354.4474182128906, 0.03939764201641083, 425.1584777832031, 354.4474182128906, 0.02348250150680542, 425.1584777832031, 354.4474182128906, 0.03718782961368561], 'score': 0.9582511186599731}
#python3.6 inspect_json.py -file /nfs/data/coco_17/annotations/person_keypoints_val2017.json -mode coco-annotations -search 163682

#Input file format:
#List of dict elements with fields:
#   - image_id
#   - category_id: not used
#   - bbox: not used
#   - keypoints
#   - score

_MODES = ['JcJLdLLa_reduced', 'JLd_all_direct', 'JJo_reduced', 'Jc_rel']
_REFs = {5: "left_shoulder", 6: "right_shoulder"} #{1: "left_shoulder"}
_NUMKPTS = 17

_FILTER = True
_MINKPTs = 5#7#5 #7#10
_KPTS_THRESHOLD = 0.05 #same as in detectron2/utils/visualizer.py

_NORM = True

_DEBUG = False
if not os.path.isfile(args.inputFile):
    raise ValueError("No valid input file.")
if _DEBUG:
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')
if args.mode not in _MODES:
    raise ValueError("No valid mode number.")
if args.target not in ['query', 'insert', 'eval']:
    raise ValueError("No valid purpose.")

_ADDNOTES = ''
_NOTESFLAG = True

def main():
    #calculateGPD(test_entry['keypoints'])
    #exit(1)
    output_dir = os.path.join('/home/althausc/master_thesis_impl/posedescriptors/out', args.target)
    output_dir = os.path.join(output_dir, datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        raise ValueError("Output directory %s already exists."%output_dir)

    print("Reading from file: ",args.inputFile)
    with open (args.inputFile, "r") as f:
        json_data = json.load(f)
    
    if args.gtAnn:
        json_data = json_data['annotations']
    json_data = sorted(json_data, key=lambda k: k['image_id']) 

    #Flip keypoints vertically to add an additional descriptor, only for target = query
    if args.flip and args.target == 'query':
        addflippedpredictions(json_data)

    #Output format: [{image_id, [gpd1,...,gpdn]}, ... ,{image_id, [gpd1,...,gpdm]}]
    json_out = []
    prevImgId = None
    #json_data = [test_entry for i in range(0,100)]
    start_time = time.time()
    
    fc = 0 #filter counter
    c = 0
    i = 0
    for i,person in enumerate(json_data):
        if "keypoints" in person:
            #Decide based on confidence score weather to process this prediction
            if not filterKeypoints(person['keypoints'], args.mode):
                fc = fc + 1
                continue

            #if person['score'] >= 0.9:
            #    print(person['keypoints'][2::3], isvalid, person['score'])
            #logging.debug("LATER:",person['keypoints'])
            metadata = {'imagesize': person['image_size'], 'bbox' : person['bbox']}
            keypoint_descriptor, confidences, mask = calculateGPD(person['keypoints'], args.mode, metadata)
            #"image_size": [448, 815], "category_id": 1, "bbox": [294.5003662109375, 21.756784439086914, 209.2066650390625, 417.60650634765625]
            barea = calculate_personpercentage(metadata)
            if args.gtAnn:
                json_out.append({"image_id": person["image_id"], "gpd": keypoint_descriptor, "mask": mask, 'score': [1]*len(keypoint_descriptor), 'confidences': confidences , 'percimage': barea})
            else:
                json_out.append({"image_id": person["image_id"], "gpd": keypoint_descriptor, "mask": mask, 'score': person['score'], 'confidences': confidences, 'percimage': barea})
            c = c + 1
        if i%1000 == 0 and i!=0:
            print("Processed %d elements."%i)

    print("Time for calculating %d descriptors = %s seconds " % (c,time.time() - start_time))
    print("Original number of json predictions: ",len(json_data))
    print("Number of calculated descriptors: ",c)

    numimages = len(set([item['image_id'] for item in json_data]))
    numimagesres = len(set([item['image_id'] for item in json_out]))

    #Apply Principal Component Analysis either from scratch or trained model for 
    #dimension reduction of computed GPDs.
    if args.pca is not None:
        model = applyPCA(json_out, dim=args.pca)
        pickle.dump(model, open(os.path.join(output_dir,'modelpca%d'%args.pca + '.pkl'), "wb"))
    if args.pcamodel is not None:
        pca_reload = pickle.load(open(args.pcamodel,'rb'))
        _ = applyPCA(json_out, pca=pca_reload)

    #Save person frequency/imageid in seperate file
    imgfreqs = getimgfrequencies(json_out)
    with open(os.path.join(output_dir, 'persons-per-image.json'), 'w') as f:
        json.dump(imgfreqs, f)

    #Writing config to file
    with open(os.path.join(output_dir, 'config.txt'), 'a') as f:
        f.write("Input file: %s"%args.inputFile + os.linesep)
        f.write("Minimum KPTS: %d"%_MINKPTs + os.linesep)
        f.write("Mode: %s"%args.mode + os.linesep)
        f.write("Filter: %d"%_FILTER + os.linesep)
        f.write("Keypoint threshold: %d"%_KPTS_THRESHOLD + os.linesep)
        f.write("Ref(s): %s"%str(_REFs) + os.linesep)
        f.write("Normalization: %s"%str(_NORM) + os.linesep)
        f.write("PCA dimension: %s"%(str(args.pca) if args.pca is not None else 'not used')+ os.linesep + os.linesep)
        
        f.write("Number input predictions: %d"%len(json_data) + os.linesep)
        f.write("Number calculated descriptors: %d"%c + os.linesep)
        f.write("Number of input prediction filtered out: %d"%fc + os.linesep)
        f.write("Number of unique input images (w.r. to predictions): %d"%numimages + os.linesep)
        f.write("Number of unique output images (w.r. to filtered predictions): %d"%numimagesres + os.linesep + os.linesep)

        f.write("Dimension of descriptor: %d"%(len(json_out[0]['gpd']) if len(json_out)>=1 else 0) + os.linesep) 
        if args.flip:
            f.write("Flip: %s"%args.flip + os.linesep)
        f.write(_ADDNOTES)

    json_file = 'geometric_pose_descriptor_c_%d_m%s_t%.2f_f%d_mkpt%dn%d'%(c,args.mode, _KPTS_THRESHOLD, _FILTER, _MINKPTs, _NORM)
    with open(os.path.join(output_dir, json_file+'.json'), 'w') as f:
        print("Writing to folder: ",output_dir)
        json.dump(json_out, f)

_BODY_PART_MAPPING = {
        0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear", 5: "left_shoulder", 6: "right_shoulder",
        7: "left_elbow", 8: "right_elbow", 9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
        13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"}

_INV_BODY_PART_MAPPING = dict((v,k) for k, v in _BODY_PART_MAPPING.items())  
# Pairs of keypoints that should be exchanged under horizontal flipping
_KPT_SYMMETRY = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)] #not used

#Keypoint connections in accordance to https://images4.programmersought.com/935/c3/c3a73bf51c47252f4a33566327e30a87.png
#Modified/Added Lines: - left-right shoulder
#                      - left hipbone-left shoulder
#                      - right hipbone-right shoulder  
#                      - left-right hipbone  
_KPTS_LINES = [(0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15)]   
_KPTS_LINES.extend([(6,5), (6,12), (5,11), (12,11)]) 

_END_JOINTS = {3: 'left_ear', 4: 'right_ear', 9: 'left_wrist', 10: 'right_wrist', 15: 'left_ankle', 16: 'right_ankle'}
_END_JOINTS_DEPTH_2 = [(0,4), (0,3), (6,10), (5,9), (12,16), (11,15)]


def calculateGPD(keypoints, mode, metadata):
    #Get the geometric pose descriptor based on:
    #   -selected mode
    #   -slected reference point(s)
    xs = keypoints[::3]
    ys = keypoints[1::3]
    cs = keypoints[2::3]

    kpts_valid = [False if x==-1 or y==-1 else True for x,y in zip(xs,ys)] 
    logging.debug("kpts_valid: {}".format(kpts_valid))

    #Delete background label if present & included in keypoint list 
    if len(xs) == len(_BODY_PART_MAPPING) and len(ys) == len(_BODY_PART_MAPPING):
        if 'Background' in _INV_BODY_PART_MAPPING:
            del xs[_INV_BODY_PART_MAPPING['Background']]
            del ys[_INV_BODY_PART_MAPPING['Background']]
            print("Info: Deleted background label.")
   
    keypoints = np.asarray(list(zip(xs,ys)), np.float32)

    # Construct different connection lines from keypoints
    l_direct_adjacent = lines_direct_adjacent(keypoints)
    l_end_depth2 = lines_endjoints_depth2(keypoints)
    l_endjoints = lines_endjoints(keypoints)
    l_custom = lines_custom(keypoints)
    #Merge all lines
    l_all = {}
    l_all.update(l_direct_adjacent)
    l_all.update(l_end_depth2)
    l_all.update(l_endjoints)
    l_all.update(l_custom)
    
    if _DEBUG:
        print("l_direct_adjacent:", l_direct_adjacent)
        print("l_end_depth2:", l_end_depth2)
        print("line_endjoints:",l_endjoints)
        print("l_custom:", l_custom)
        print("l_custom_all:",len(l_all))

    pose_descriptor = []
    global _NOTESFLAG
    global _ADDNOTES
    
    if mode == 'JcJLdLLa_reduced':
        #Descriptor contains all joint coordinates, selected joint-line distances and selected line-line angles 
        #Result dimension: 64

        #Dimensions: 17 keypoints, 17x2 values (with or without visibility flag)
        #            + 1 reference keypoint, 1x2 values
        joint_coordinates = joint_coordinates_rel(keypoints, kpts_valid, metadata['imagesize'], addconfidences=None)
        #joint_coordinates = joint_coordinates_rel(keypoints, ref_point.tolist(), visiblities = vs , vclipping = True)
        pose_descriptor.append(joint_coordinates)
        if _NOTESFLAG:
            _ADDNOTES += 'Dimensions JCoords %d \n'%len(joint_coordinates)

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
        JL_d = joint_line_distances(keypoints, l_all, kpts_valid, kpt_line_mapping)
        pose_descriptor.append(JL_d)

        if _NOTESFLAG:
            _ADDNOTES += 'Dimensions JL_d: %d \n'%len(JL_d)

        #Dimensions: 12 angles
        line_line_mapping = {(10,9):[[(9,15),'rhand_lhandrfoot'],[(9,16),'rhand_lhandlfoot'],
                                    [(10,16),'lhand_rhandlfoot'],[(10,15),'lhand_rhandrfoot']],
                            (5,11):[(5,9),'hand_shoulder_hip_l'], (6,12):[(6,10),'hand_shoulder_hip_r'],
                            (6,8):[(5,7),'upper_arms'], (8,10):[(7,9),'lower_arms'],
                            (12,14):[(11,13),'upper_legs'], (14,16):[(13,15),'lower_legs'],
                            (0,5):[(3,5),'head_shoulder_l'], (0,6):[(4,6),'head_shoulder_r']}
        LL_a = line_line_angles(l_all, kpts_valid, line_line_mapping)
        
        if _NOTESFLAG:
            _ADDNOTES += 'Dimensions LL_a: %d \n'%len(LL_a)

        pose_descriptor.append(LL_a)

    elif mode == 'JLd_all_direct':
        #Descriptor contains all joint-line distances
        #Specify which lines should be used, either: 
        #   - all lines: Z * 17 keypoints = ..
        #   - adjacent lines: 14*17 = 238
        #   - endjoint connection lines: X*17 = ..
        #   - endjoint depth 2 connection lines: Y*17 = ..

        JL_d = joint_line_distances(keypoints, l_direct_adjacent, kpts_valid)
        JL_d = [entry for k,entry in enumerate(JL_d) if k%2 == 0]
        pose_descriptor.append(JL_d)

        if _NOTESFLAG:
            _ADDNOTES += 'Dimensions JL_d: %d \n'%len(JL_d)

    elif mode == 'JJo_reduced':
        #Descriptor contains normalized joint-joint orientations
        #Used: all limbs in COCO keypoint model (direction facing outwards) = 16 
        #       + lines between endjoints and depth 2 from endjoints = 6
        #       + custom lines
        custom_jj_mapping = [(9,15), (10,16), #endjoints
                             (0,6), (0,5)] #head-shoulders
        kpt_kpt_mapping = []
        kpt_kpt_mapping.extend(_KPTS_LINES)
        kpt_kpt_mapping.extend(_END_JOINTS_DEPTH_2)
        kpt_kpt_mapping.extend(custom_jj_mapping)
        
        JJ_o = joint_joint_orientations(keypoints, kpts_valid, kpt_kpt_mapping)
        pose_descriptor.append(JJ_o)

        if _NOTESFLAG:
            _ADDNOTES += 'Dimensions JJ_o: %d \n'%len(JJ_o)

    elif mode == 'Jc_rel':
        Jc_rel = joint_coordinates_rel(keypoints, kpts_valid, metadata['imagesize'], addconfidences=None)
        pose_descriptor.append(Jc_rel)


    #indices_pairs = []
    #JJ_d = joint_joint_distances(keypoints,indices_pairs=None)
    #pose_descriptor.append(JJ_d)
    
    #indices_pairs = []
    #JJ_o = joint_joint_orientations(keypoints, indices_pairs=None)
    #pose_descriptor.append(JJ_o)

    #Add clipped score value
    #score = max(0,min(1,score))
    #pose_descriptor.append([score])

    #Planes for selected regions: 2 for each head, arms, leg & foot region 
    #plane_points = [{1: 'Neck',0: 'Nose', 18: 'LEar'}, {1: 'Neck',0: 'Nose', 18: 'REar'}, {2: 'RShoulder', 3: 'RElbow', 4: 'RWrist'},
    #                {5: 'LShoulder', 6: 'LElbow',7: 'LWrist'}, {12: 'LHip', 13: 'LKnee', 14: 'LAnkle'}, { 9: 'RHip', 10: 'RKnee', 11: 'RAnkle'},
    #                {13: 'LKnee', 14: 'LAnkle', 19: 'LBigToe'}, {10: 'RKnee', 11: 'RAnkle', 22: 'RBigToe'}]

    #planes = get_planes(plane_points)
    #JP_d = joint_plane_distances(keypoints, planes)
    if _DEBUG:
        print(pose_descriptor)
    pose_descriptor = [item for sublist in pose_descriptor for item in sublist]
    for v in pose_descriptor:
        if math.isnan(v):
            print("Detected NaN value in final descriptor: ")
            print(pose_descriptor)
            exit(1)
    #logging.debug(pose_descriptor)
    logging.debug("\nDimension of pose descriptor: {}".format(len(pose_descriptor)) ) 
    #flatten desriptor
    
    logging.debug("Dimension of pose descriptor flattened: {}".format(len(pose_descriptor)))
    logging.debug("\n")

    if _NOTESFLAG:
        _ADDNOTES += 'Lines directly adjacent: %d \n'%len(l_direct_adjacent)
        _ADDNOTES += 'Lines endjoints depth 2: %d \n'%len(l_end_depth2)
        _ADDNOTES += 'Lines endjoint connections: %d \n'%len(l_endjoints)
        _ADDNOTES += 'Lines custom: %d \n'%len(l_custom)
        _ADDNOTES += 'Lines together: %d \n'%len(l_all)

        _NOTESFLAG = False

    mask = ''.join(['1' if entry!=-1 else '0' for entry in pose_descriptor])

    return pose_descriptor, cs, mask

def calculate_personpercentage(metadata):
    #Calculates the proportion of the persons bbox w.r. to the total image area
    imgarea = metadata['imagesize'][0] * metadata['imagesize'][1]
    #personarea =  abs(metadata['bbox'][0] - metadata['bbox'][2]) * abs(metadata['bbox'][1] - metadata['bbox'][3]) #XYXY
    personarea =  metadata['bbox'][2] * metadata['bbox'][3] #XYWH
    return float(personarea/imgarea)

# ------------------------------------- DESCRIPTORS OF DIFFERENT TYPES ------------------------------------- 

def joint_coordinates_rel(keypoints, kptsvalid, imagesize, addconfidences = None):
    #Joint Coordinate: all keypoints in relative coordinate system + reference point in world coordinate system
    # Make all keypoints relative to a selected reference point
    ref_ids = list(_REFs.keys())
    if len(_REFs) == 2:
        p1 = keypoints[ref_ids[0]]
        p2 = keypoints[ref_ids[1]]
        pmid = [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2]
    elif len(_REFs) == 1:
        pmid = keypoints[ref_ids[0]]
    else:
        raise ValueError("_REFS not valid.")

    ref_point = np.array(pmid)
    keypoints = (keypoints - ref_point).tolist()

    #Dimension 17 x 2 + 2(reference point) [+ 17 (confidences)]
    joint_coordinates = []
    for i,k in enumerate(keypoints):
        if not kptsvalid[i]:
            joint_coordinates.extend([-1,-1])
        else:
            joint_coordinates.extend(k)
    #Normalization allows for scale invariance
    #Only normalize valid entries, since -1 entries would falsify the normalization result  
    if _NORM:      
        joint_coordinates = normalizevec(joint_coordinates, mask=True)
    
    #Add relative position of pose to descriptor
    height, width = imagesize[0], imagesize[1]
    relative_refpoint = [pmid[0]/width, pmid[1]/height]
    #print(pmid[0], width, pmid[1], height ,pmid[0]/width, pmid[1]/height)
    joint_coordinates.extend(relative_refpoint)

    if addconfidences is not None:
        #Clip values bcose sometime very large
        #Lower clip value if too much effect on search result
        confs = map(lambda y: max(y,1), addconfidences)
        joint_coordinates.extend(confs)
    if _DEBUG:
        print(joint_coordinates)
    logging.debug("Dimension joint coordinates: {}".format(len(joint_coordinates))) 
    return joint_coordinates

def joint_joint_distances(keypoints, kptsvalid, indices_pairs=None):
    #Joint-Joint Distance
    joint_distances = []
    if indices_pairs is None:
        #Dimension 25 over 2 = 300 / 17 over 2 = 136
        #scipy.special.binom(len(keypoints), 2))
        #Only consider unique different joint pairs ,e.g. (1,2) == (2,1)
        for i1 in range(len(keypoints)):
            for i2 in range(len(keypoints)):
                if i1 < i2:
                    if not kptsvalid[i1] or not kptsvalid[i2]:
                        joint_distances.append(-1)
                    else:    
                        joint_distances.append(np.linalg.norm(keypoints[i2]-keypoints[i1]))
    else:
        for start,end in indices_pairs:
            if not kptsvalid[start] or not kptsvalid[end]:
                joint_distances.append(-1)
            else: 
                joint_distances.append(np.linalg.norm(keypoints[start]-keypoints[end]))
    
    logging.debug("Dimension joint distances: {}".format(len(joint_distances))) 
    if _NORM:
        joint_distances = normalizevec(joint_distances, mask=True)
    return joint_distances

def joint_joint_orientations(keypoints, kptsvalid, indices_pairs=None):
    #Joint-Joint Orientation (vector orientation of unit vector from j1 to j2)
    joint_orientations = []
    if indices_pairs is None:
        #Dimension 25 over 2 = 300 / 17 over 2 = 136
        for i1 in range(len(keypoints)):
            for i2 in range(len(keypoints)):
                if i1 < i2:
                    j1 = keypoints[i1]
                    j2 = keypoints[i2]
                    #Don't compute for unvalid points or points with same coordinates
                    if (j1==j2).all():
                        joint_orientations.extend([0,0])
                    elif not kptsvalid[i1] or not kptsvalid[i2]:
                        joint_orientations.extend([-1,-1])
                    else:    
                        vec = np.subtract(j2,j1)
                        normvec = vec/np.linalg.norm(vec)
                        normvec = normvec.astype(float)
                        joint_orientations.extend(list(normvec))

    else:
        for start,end in indices_pairs:
            j1 = keypoints[start]
            j2 = keypoints[end]
            #Don't compute for unvalid points or points with same coordinates
            if (j1==j2).all():
                joint_orientations.extend([0,0])
            elif not kptsvalid[start] or not kptsvalid[end]:
                joint_orientations.extend([-1,-1])
            else:   
                vec = np.subtract(j2,j1)
                normvec = vec/np.linalg.norm(vec)
                normvec = normvec.astype(float)
                #print("vec: ",vec)
                #print("normvec: ",normvec)
                joint_orientations.extend(list(normvec))
                #plt.axis('equal')
                #plt.plot([j1[0], j2[0]], [j1[1], j2[1]], 'k-', lw=1)
                #plt.plot([0, vec[0]], [0, vec[1]], 'g-', lw=1)
                #plt.plot([0, normvec[0]], [0, normvec[1]], 'r-', lw=1)
                #plt.gca().invert_yaxis()
                #print("(%s,%s)->(%s,%s)"%(_BODY_PART_MAPPING[k11], _BODY_PART_MAPPING[k12], _BODY_PART_MAPPING[k21], _BODY_PART_MAPPING[k22]))
                #label = "%s-%s"%(_BODY_PART_MAPPING[start], _BODY_PART_MAPPING[end])
                #print(label)
                #plt.savefig("/home/althausc/master_thesis_impl/posedescriptors/out/query/11-09_12-42-08/.test/%s.jpg"%label)
                #plt.clf()

            #plt.gca().invert_yaxis()
            #plt.savefig("/home/althausc/master_thesis_impl/posedescriptors/out/query/11-09_12-42-08/.test/entirebody.jpg")       
    logging.debug("Dimension of joint orientations: {}".format(len(joint_orientations)))
    #if _NORM:
    #    joint_orientations = normalizevec(joint_orientations, mask=True)
    return joint_orientations 

def joint_line_distances(keypoints, lines, kptsvalid, kpt_line_mapping = None):
    #Joint-Line Distance
    #Modes: 1. calculate distance between lines and all keypoints (kpt_line_mapping = None)
    #       2. calculate distance only for specified lines and specified keypoints (kpt_line_mappints != None)
    joint_line_distances = []
    
    if kpt_line_mapping is None:
        #Approx. Dimension: 60 lines * (25-3) joints = 1320 / (16+6+15) lines * (17-3) joints = 518
        #print("lines.keys(): ", lines.keys())
        #print("keypoints: ", keypoints)
        for k, l in lines.items():
            coords = list(l.coords)
            #print("coords: ", coords)
            #print(k)
            for i,joint in enumerate(keypoints):
                #if joint is the same as either start or end point of the line, continue
                if i in k: 
                    continue
                if not kptsvalid[i] or not kptsvalid[k[0]] or not kptsvalid[k[1]]:
                    joint_line_distances.append(-1)
                    #print('%s->(%s,%s) %f'%(_BODY_PART_MAPPING[i],_BODY_PART_MAPPING[k[0]],_BODY_PART_MAPPING[k[1], -1]))
                else:
                    joint_line_distances.append(Point(joint).distance(l))  
                    #print('%s->(%s,%s) %f'%(_BODY_PART_MAPPING[i],_BODY_PART_MAPPING[k[0]],_BODY_PART_MAPPING[k[1]], Point(joint).distance(l)))         
    else:
        for k, [(k1,k2),label] in dict_to_item_list(kpt_line_mapping):
            if _DEBUG:
                print('%s->(%s,%s)'%(_BODY_PART_MAPPING[k],_BODY_PART_MAPPING[k1],_BODY_PART_MAPPING[k2]))

            if not kptsvalid[k] or not kptsvalid[k1] or not kptsvalid[k2]:
                joint_line_distances.append(-1)
                continue                
            if (k1,k2) in lines.keys():
                joint_line_distances.append(Point(keypoints[k]).distance(lines[(k1,k2)]))
                #print(Point(keypoints[k]).distance(lines[(k1,k2)]))
            elif (k2,k1) in lines.keys():
                joint_line_distances.append(Point(keypoints[k]).distance(lines[(k2,k1)]))
                #print(Point(keypoints[k]).distance(lines[(k2,k1)]))
            else:
                logging.debug("Not found line: {}{}".format(k1,k2))
                print("Not found")
   
    logging.debug("Dimension joint line distances: {}".format(len(joint_line_distances)))
    if _NORM:
        joint_line_distances = normalizevec(joint_line_distances, mask=True)
    return joint_line_distances

def line_line_angles(lines, kptsvalid, line_line_mapping = None):
    #Line-Line Angle
    line_line_angles = []    
    
    if line_line_mapping is None:
        #(25-1) over 2 = 276 / (17-1) over 2 = 120
        finished = []
        for (k11,k12),l1 in lines.items():
            for (k21,k22),l2 in lines.items():
                if not kptsvalid[k11] or not kptsvalid[k12] or not kptsvalid[k21] or not kptsvalid[k22]:
                    line_line_angles.append(-1)
                    continue
                #skip self-angle and already calculated angles of same lines
                if(k21,k22) == (k11,k12) or [(k11,k12),(k21,k22)] in finished or [(k21,k22),(k11,k12)] in finished:
                    continue
                line_line_angles.append(angle(l1,l2))
                finished.append([(k11,k12),(k21,k22)])
    else:
        k = 0
        for (k11,k12),[(k21,k22),label] in dict_to_item_list(line_line_mapping):
            if not kptsvalid[k11] or not kptsvalid[k12] or not kptsvalid[k21] or not kptsvalid[k22]:
                line_line_angles.append(-1)
                continue
            l1 = lines[(k11,k12)] if (k11,k12) in lines.keys() else lines[(k12,k11)]
            l2 = lines[(k21,k22)] if (k21,k22) in lines.keys() else lines[(k22,k21)]
            #an= angle(l1,l2)
            #print(k,"angle between ", l1,l2, "is", an, math.degrees(an))
            #print("(%s,%s)->(%s,%s)"%(_BODY_PART_MAPPING[k11], _BODY_PART_MAPPING[k12], _BODY_PART_MAPPING[k21], _BODY_PART_MAPPING[k22]))
            #[p11,p12] = list(l1.coords)
            #[p21,p22] = list(l2.coords)
            #plt.axis('equal')
            #plt.plot([p11[0], p12[0]], [p11[1], p12[1]], 'k-', lw=1)
            #plt.plot([p21[0], p22[0]], [p21[1], p22[1]], 'k-', lw=1)
            #print(k,"angle between ", l1,l2, "is", an, math.degrees(an))
            #print((k11,k12), (k21,k22), (k11,k12) in lines.keys(), (k21,k22) in lines.keys())
            line_line_angles.append(angle(l1,l2))
            #plt.gca().invert_yaxis()
            #plt.savefig("/home/althausc/master_thesis_impl/posedescriptors/out/query/11-09_12-42-08/.test/angle%d.jpg"%k)
            #plt.clf()
            #k += 1

    logging.debug("Dimension line line angles: {}".format(len(line_line_angles)) )
    #if _NORM: #not reasonable for angle vector
    #    #line_line_angles = normalizevec(line_line_angles, rangemin=0, rangemax=math.pi)
    #    line_line_angles = normalizevec(line_line_angles, mask=True)
    return line_line_angles

def joint_plane_distances(keypoints,planes):
    #Joint-Plane Distance
    #Dimensions (25-3)*8 = 176
    joint_plane_distances = []
    for joint in keypoints:
        for plane in planes:
            if tuple(joint) not in list(plane.exterior.coords):
                joint_plane_distances.append(Point(joint).distance(plane))
            #To provide static descriptor dimension
            else:
                joint_plane_distances.append(0)
    logging.debug("Dimension joint plane distances: {}".format(len(joint_plane_distances))) 
    return normalizevec(joint_plane_distances)


# ------------------------------------ HELPER FUNCTIONS -----------------------------------
def filterKeypoints(pose_keypoints, mode):
    #Filter out keypoints above a treshold
    #Skip pose if too few keypoints or reference point is not contained (for keypoint coordinate descriptor)
    #Probability = confidence score for the keypoint (not visibility flag!)
    
    if mode == 'JcJLdLLa_reduced': #TODO: plus Jc_rel?
        for ref in _REFs.keys():
            if pose_keypoints[ref*3+2] <= _KPTS_THRESHOLD:
                return False  
    c = 0
    for idx in range(0,_NUMKPTS):
        x, y, prob = pose_keypoints[idx*3:idx*3+3]
        #prob: 0 (no keypoint) or low value (not sure)
        if prob <= _KPTS_THRESHOLD: 
            pose_keypoints[idx*3:idx*3+3] = [-1,-1,-1]
        else:
            c = c + 1
    logging.debug("Not valid keypoints: {}".format(_NUMKPTS-c))
    if c >= _MINKPTs:
        return True
    else:
        return False

def addflippedpredictions(preds):
    print("Adding flipped keypoint predictions ...")
    c_flip = 0
    predscopy = copy.deepcopy(preds)
    for i,pred in enumerate(predscopy):
        refs = _REFs.keys()
        #supports more than 2 keypoints
        refpointx = sum([ pred['keypoints'][k*3] for k in refs])/len(refs)
        refpointy = sum([ pred['keypoints'][k*3+1] for k in refs])/len(refs)
        for k in range(0,len(pred['keypoints']), 3):
            if pred['keypoints'][k]> refpointx:
                 pred['keypoints'][k] = pred['keypoints'][k] - refpointx
            else:
                 pred['keypoints'][k] = refpointx - pred['keypoints'][k]

            if pred['keypoints'][k+1]> refpointy:
                 pred['keypoints'][k+1] = pred['keypoints'][k+1] - refpointy
            else:
                 pred['keypoints'][k+1] = refpointy - pred['keypoints'][k+1] 
        pred['image_id'] = '{}-flipped'.format(pred['image_id']) 
        preds.append(pred)
        c_flip = c_flip + 1
    print("Added {} flipped keypoint predictions".format(c_flip))

def lines_direct_adjacent(keypoints):
    #Directly adjacent lines (linemapping of _KPTS_LINES), Dimesions: 25 - 1 / 17-1 = 16
    #Lines pointing outwards for ordering
    lines = {}
    for start,end in _KPTS_LINES:
        lines.update({(start,end) : LineString([keypoints[start], keypoints[end]])})
    logging.debug("\tDimension lines between directly adjacent keypoints: {}".format(len(lines)))
    return lines

def lines_endjoints_depth2(keypoints):
    #Kinetic chain from end joints of depth 2
    #Lines pointing outwards for ordering
    #Dimension (BODY_MODEL_25): 8 / 6
    lines = {}
    for (begin, end) in _END_JOINTS_DEPTH_2:
        lines.update({(begin,end) : LineString([keypoints[begin], keypoints[end]])})
    logging.debug("\tDimension lines from end-joints of depth 2: {}".format(len(lines)))
    return lines

def lines_endjoints(keypoints, indices_pairs = None):
    #Lines only between end joints
    #Lines starting with highest point for ordering
    # when no such, then from left to right
    lines = {}
    if indices_pairs is None:
        #Dimensions: num(_END_JOINTS) over 2 , BODY_MODEL_25: 8 over 2 = 28 / 6 over 2 = 15
        #Add left-to right of same height first
        for (k1,k2) in [(4,3), (10,9), (16,15)]:
            lines.update({(k1,k2) : LineString([keypoints[k1], keypoints[k2]])})
        for (k1,label1) in _END_JOINTS.items():
            for (k2,label2) in _END_JOINTS.items():
                if k1!=k2 and (k2,k1) not in lines and (k1,k2) not in lines:
                    lines.update({(k1,k2) : LineString([keypoints[k1], keypoints[k2]])})
    else:
        for start,end in indices_pairs:
            if start in _END_JOINTS.keys() and end in _END_JOINTS.keys():
                lines.update({(start,end) : LineString([keypoints[start], keypoints[end]])})
            else:
                raise ValueError("Given index is no end joint.")
    logging.debug("\tDimension lines between end-joints only: {}".format(len(lines))) 
    return lines

def lines_custom(keypoints):
    #Lines which are considered important and not computed by other line methods,
    #Important: some lines are needed for angle computations
    #Lines starting with highest point for ordering
    #when no such, then from left to right
    kpt_mapping = {14:15, 13:16, #foot knees x
                    0:[5,6],  #shoulders-nose
                    5:8, 6:7, #elbows-shoulders x
                    3:5, 4:6  #shoulders-ears
                    }
    lines = {}
    for k1,k2 in dict_to_item_list(kpt_mapping, level=0):
        lines.update({(k1,k2) : LineString([keypoints[k1], keypoints[k2]])})
    logging.debug("\tDimension lines custom: {}".format(len(lines))) 
    return lines

def get_planes(plane_points):
    planes = []
    for p in plane_points:
        indices = list(p.keys())
        planes.append(Polygon([ keypoints[indices[0]],keypoints[indices[1]],keypoints[indices[2]] ]))
    return planes

if __name__=="__main__":
   main()





    





