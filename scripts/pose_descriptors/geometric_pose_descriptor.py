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
import matplotlib.pyplot as plt
from utils import applyPCA, normalizevec, dict_to_item_list, line_intersection, angle

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
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")
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

_VISIBILITY_TRESH = 0.1 #refering to visibility probability of keypoints , TODO:check if valid, range [0,1]? 
_REFs = {5: "left_shoulder", 6: "right_shoulder"} #{1: "left_shoulder"}
_MINKPTs = 10
_NUMKPTS = 17
_MODES = ['JcJLdLLa_reduced', 'JLd_all_direct']
_FILTER = False#True

if not os.path.isfile(args.inputFile):
    raise ValueError("No valid input file.")
if args.verbose:
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')
if args.mode not in _MODES:
    raise ValueError("No valid mode number.")
if args.target not in ['query', 'insert']:
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

    #Output format: [{image_id, [gpd1,...,gpdn]}, ... ,{image_id, [gpd1,...,gpdm]}]
    json_out = []
    prevImgId = None
    #json_data = [test_entry for i in range(0,100)]
    start_time = time.time()

    c = 0
    i = 0
    for i,person in enumerate(json_data):
        if "keypoints" in person:
            #logging.debug("PREV:",person['keypoints'])
            isvalid = True
            if _FILTER:
                #Decide based on visibility score weather to process this prediction
                isvalid = filterKeypoints(person['keypoints'])
            #if person['score'] >= 0.9:
            #    print(person['keypoints'][2::3], isvalid, person['score'])
            #logging.debug("LATER:",person['keypoints'])

            if isvalid:
                keypoint_descriptor, visibilities = calculateGPD(person['keypoints'], args.mode)
                if args.gtAnn:
                    json_out.append({"image_id": person["image_id"], "gpd": keypoint_descriptor, 'score': [1]*len(keypoint_descriptor), 'vis': visibilities})
                else:
                    json_out.append({"image_id": person["image_id"], "gpd": keypoint_descriptor, 'score': person['score'], 'vis': visibilities})
                c = c + 1
        if i%1000 == 0 and i!=0:
            print("Processed %d elements."%i)

    print("Time for calculating %d descriptors = %s seconds " % (c,time.time() - start_time))
    print("Original number of json predictions: ",len(json_data))
    print("Number of calculated descriptors: ",c)

    #Apply Principal Component Analysis either from scratch or trained model for 
    #dimension reduction of computed GPDs.
    if args.pca is not None:
        model = applyPCA(json_out, dim=args.pca)
        pickle.dump(model, open(os.path.join(output_dir,'modelpca%d'%args.pca + '.pkl'), "wb"))
    if args.pcamodel is not None:
        pca_reload = pickle.load(open(args.pcamodel,'rb'))
        _ = applyPCA(json_out, pca=pca_reload)

    #Writing config to file
    with open(os.path.join(output_dir, 'config.txt'), 'a') as f:
        f.write("Input file: %s"%args.inputFile + os.linesep)
        f.write("Minimum KPTS: %d"%_MINKPTs + os.linesep)
        f.write("Mode: %s"%args.mode + os.linesep)
        f.write("Filter: %d"%_FILTER + os.linesep)
        f.write("Keypoint threshold: %d"%_VISIBILITY_TRESH + os.linesep)
        f.write("Ref(s): %s"%str(_REFs) + os.linesep)
        f.write("PCA dimension: %s"%(str(args.pca) if args.pca is not None else 'not used')+ os.linesep)
        f.write("Number input predictions: %d"%len(json_data) + os.linesep)
        f.write("Number calculated descriptors: %d"%c + os.linesep)
        f.write("Dimension of descriptor: %d"%len(json_out[0]['gpd']) + os.linesep)
        f.write(_ADDNOTES)

    json_file = 'geometric_pose_descriptor_c_%d_m%s_t%.2f_f%d_mkpt%d'%(c,args.mode, _VISIBILITY_TRESH, _FILTER, _MINKPTs)
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


def calculateGPD(keypoints, mode):
    #Get the geometric pose descriptor based on:
    #   -selected mode
    #   -slected reference point(s)
    xs = keypoints[::3]
    ys = keypoints[1::3]
    vs = keypoints[2::3]

    num_kps = 17
    kpts_valid = [1 if v is not None else 0 for v in vs] #Keypoints with visibility None should not be considered (0 in feature decsriptor)
    logging.debug("kpts_valid: {}".format(kpts_valid))

    #Delete background label if present & included in keypoint list 
    if len(xs) == len(_BODY_PART_MAPPING) and len(ys) == len(_BODY_PART_MAPPING):
        if 'Background' in _INV_BODY_PART_MAPPING:
            del xs[_INV_BODY_PART_MAPPING['Background']]
            del ys[_INV_BODY_PART_MAPPING['Background']]
            print("Info: Deleted background label.")
   
    # Make all keypoints relative to a selected reference point
    if len(_REFs) == 2:
        ref_ids = list(_REFs.keys())
        p1 = [xs[ref_ids[0]], ys[ref_ids[0]]]
        p2 = [xs[ref_ids[1]], ys[ref_ids[1]]]
        pmid = [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2 ]
        vs.insert(0, (vs[ref_ids[0]] + vs[ref_ids[1]])/2)
    elif len(_REFs) == 1:
        pmid = [xs[ref_ids[0]], ys[ref_ids[0]]]
        vs.insert(0, vs[ref_ids[0]])
    else:
        raise ValueError("_REFS not valid.")

    ref_point = np.array(pmid)
    keypoints = np.asarray(list(zip(xs,ys)), np.float32)
    keypoints = (keypoints - ref_point).tolist()

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
        joint_coordinates = joint_coordinates_rel(keypoints)
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
        pose_descriptor.append(JL_d)

        if _NOTESFLAG:
            _ADDNOTES += 'Dimensions JL_d: %d \n'%len(JL_d)

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

    for l in pose_descriptor:
        for v in l:
            if math.isnan(v):
                print("Detected NaN value in final descriptor: ")
                print(pose_descriptor)
                exit(1)
    #logging.debug(pose_descriptor)
    logging.debug("\nDimension of pose descriptor: {}".format(len(pose_descriptor)) ) 
    #flatten desriptor
    pose_descriptor = [item for sublist in pose_descriptor for item in sublist]
    logging.debug("Dimension of pose descriptor flattened: {}".format(len(pose_descriptor)))
    logging.debug("\n")

    #Visibilities of kpts clipped
    print("vs:",vs)
    vs = list(map(lambda x: 0 if x is None else x, vs))
    print("vs:",vs)
    visibilities = list(map(lambda y: max(0,min(2,y)), vs))
    print("visibilities:", visibilities)

    _ADDNOTES += 'Lines directly adjacent: %d \n'%len(l_direct_adjacent)
    _ADDNOTES += 'Lines endjoints depth 2: %d \n'%len(l_end_depth2)
    _ADDNOTES += 'Lines endjoint connections: %d \n'%len(l_endjoints)
    _ADDNOTES += 'Lines custom: %d \n'%len(l_custom)
    _ADDNOTES += 'Lines together: %d \n'%len(l_all)

    _NOTESFLAG = False

    return pose_descriptor, visibilities

# ------------------------------------- DESCRIPTORS OF DIFFERENT TYPES ------------------------------------- 

def joint_coordinates_rel(keypoints, visiblities = None, vclipping = False, normalize=True):
    #Joint Coordinate: all keypoints in relative coordinate system + reference point in world coordinate system
    joint_coordinates = []
    
    if visiblities is None:
        #Dimension 17 x 2 + 1 x 2 = 36 
        #joint_coordinates = np.delete(keypoints,_INV_BODY_PART_MAPPING['MidHip'])
        #joint_coordinates = itertools.chain([reference_point], keypoints)
        joint_coordinates = [item for sublist in list(keypoints) for item in sublist]
        logging.debug("Dimension joint coordinates: {}".format(len(joint_coordinates))) 

    else:
        #Adding the visibility flag
        #Meaning:   visibility == 0 that keypoint not in the image
        #           visibility == 1 that keypoint is in the image BUT not visible, 
        #           visibility == 2 that keypoint is visible
        #Dimension 17 x 3 + 1 x 3 = 54 
        #joint_coordinates = itertools.chain([reference_point], keypoints)
        if vclipping:
            #Clip values bcose sometime very large
            visiblities = map(lambda y: max(0,min(2,y)), visiblities)
        joint_coordinates = [(xy[0],xy[1],v) for [xy, v] in zip(keypoints, visiblities)]
        joint_coordinates = [item for sublist in joint_coordinates for item in sublist]
        logging.debug("Dimension joint coordinates: {}".format(len(joint_coordinates))) 

    #Normalization allows for scale invariance
    return normalizevec(joint_coordinates)

def joint_joint_distances(keypoints,indices_pairs=None):
    #Joint-Joint Distance
    joint_distances = []
    if indices_pairs is None:
        #Dimension 25 over 2 = 300 / 17 over 2 = 136
        #scipy.special.binom(len(keypoints), 2))
        #Only consider unique different joint pairs ,e.g. (1,2) == (2,1)
        for i1 in range(len(keypoints)):
            for i2 in range(len(keypoints)):
                if i1 < i2:
                    joint_distances.append(np.linalg.norm(keypoints[i2]-keypoints[i1]))
    else:
        for start,end in indices_pairs:
            joint_distances.append(np.linalg.norm(keypoints[start]-keypoints[end]))
    
    logging.debug("Dimension joint distances: {}".format(len(joint_distances))) 
    return normalizevec(joint_distances)

def joint_joint_orientations(keypoints, indices_pairs=None):
    #Joint-Joint Orientation (vector orientation of unit vector from j1 to j2)
    joint_orientations = []
    if indices_pairs is None:
        #Dimension 25 over 2 = 300 / 17 over 2 = 136
        for i1 in range(len(keypoints)):
            for i2 in range(len(keypoints)):
                if i1 < i2:
                    j1 = keypoints[i1]
                    j2 = keypoints[i2]
                    #Don't compute for zero points or points with same coordinates
                    if np.any(j1) and np.any(j2) and not (j1==j2).all():
                        n1 = ((j2-j1)/np.linalg.norm(j2-j1))[0]
                        n2 = ((j2-j1)/np.linalg.norm(j2-j1))[1]
                        joint_orientations.append( tuple((j2-j1)/np.linalg.norm(j2-j1)) ) #a-b = target-origin
                    else:
                        joint_orientations.append((0,0))
    else:
        for start,end in indices_pairs:
            j1 = keypoints[i1]
            j2 = keypoints[i2]
            #Don't compute for zero points or points with same coordinates
            if np.any(j1) and np.any(j2) and not (j1==j2).all():
                n1 = ((j2-j1)/np.linalg.norm(j2-j1))[0]
                n2 = ((j2-j1)/np.linalg.norm(j2-j1))[1]
                joint_orientations.append( tuple((j2-j1)/np.linalg.norm(j2-j1)) ) #a-b = target-origin
            else:
                joint_orientations.append((0,0))
    logging.debug("Dimension of joint orientations: {}".format(len(joint_orientations)))
    return joint_orientations #TODO:check for normalization

def joint_line_distances(keypoints, lines, kpts_valid, kpt_line_mapping = None):
    #Joint-Line Distance
    #Modes: 1. calculate distance between lines and all keypoints (kpt_line_mapping = None)
    #       2. calculate distance only for specified lines and specified keypoints (kpt_line_mappints != None)
    joint_line_distances = []
    
    if kpt_line_mapping is None:
        #Approx. Dimension: 60 lines * (25-3) joints = 1320 / (16+6+15) lines * (17-3) joints = 518
        for k, l in lines.items():
            coords = list(l.coords)
            for i,joint in enumerate(keypoints):
                #if joint is the same as either start or end point of the line, continue
                if i in k: 
                    continue
                if not kpts_valid[i]:
                    joint_line_distances.append(0)
                    continue
                joint_line_distances.append(Point(joint).distance(l))
                
    else:
        for k, [(k1,k2),label] in dict_to_item_list(kpt_line_mapping):
            print('%s->(%s,%s)'%(_BODY_PART_MAPPING[k],_BODY_PART_MAPPING[k1],_BODY_PART_MAPPING[k2]))
            if not kpts_valid[k1] or not kpts_valid[k2]:
                joint_line_distances.append(0)
                continue                
            if (k1,k2) in lines.keys():
                joint_line_distances.append(Point(keypoints[k]).distance(lines[(k1,k2)]))
                print(Point(keypoints[k]).distance(lines[(k1,k2)]))
            elif (k2,k1) in lines.keys():
                joint_line_distances.append(Point(keypoints[k]).distance(lines[(k2,k1)]))
                print(Point(keypoints[k]).distance(lines[(k2,k1)]))
            else:
                logging.debug("Not found line: {}{}".format(k1,k2))
                print("Not found")
   
    logging.debug("Dimension joint line distances: {}".format(len(joint_line_distances)))
    return joint_line_distances
    #return normalizevec(joint_line_distances)

def line_line_angles(lines, kpts_valid, line_line_mapping = None):
    #Line-Line Angle
    line_line_angles = []    
    

    if line_line_mapping is None:
        #(25-1) over 2 = 276 / (17-1) over 2 = 120
        finished = []
        for (k11,k12),l1 in lines.items():
            for (k21,k22),l2 in lines.items():
                if not kpts_valid[k11] or not kpts_valid[k12] or not kpts_valid[k21] or not kpts_valid[k22]:
                    line_line_angles.append(0)
                    continue
                #skip self-angle and already calculated angles of same lines
                if(k21,k22) == (k11,k12) or [(k11,k12),(k21,k22)] in finished or [(k21,k22),(k11,k12)] in finished:
                    continue
                line_line_angles.append(angle(l1,l2))
                finished.append([(k11,k12),(k21,k22)])
                        #don't consider origin(zero-point)
                        #if np.any(j1) and np.any(j2): 
                        #else:
                        #    line_line_angles.append(0)
                    #To provide static descriptor dimension
                    #else:
                    #    line_line_angles.append(0)
    else:
        k = 0
        for (k11,k12),[(k21,k22),label] in dict_to_item_list(line_line_mapping):
            if not kpts_valid[k11] or not kpts_valid[k12] or not kpts_valid[k21] or not kpts_valid[k22]:
                line_line_angles.append(0)
                continue
            l1 = lines[(k11,k12)] if (k11,k12) in lines.keys() else lines[(k12,k11)]
            l2 = lines[(k21,k22)] if (k21,k22) in lines.keys() else lines[(k22,k21)]
            if l1!=l2:
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
            else:
                #lines having the exact same directions
                line_line_angles.append(0.0)
    exit(1)
    logging.debug("Dimension line line angles: {}".format(len(line_line_angles)) )
    return line_line_angles
    #return normalizevec(line_line_angles)

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

def filterKeypoints(pose_keypoints):
    #Filter out keypoints above a treshold
    #Skip pose if too few keypoints or reference point is not contained
    
    #Visibility flag: v=0: not labeled (in which case x=y=0), 
    #                 v=1: labeled but not visible
    #                 v=2: labeled and visible 

    #Keypoints are sorted out or given back fully/not reduced
        c = 0
        for idx in range(0,_NUMKPTS):
            x, y, prob = pose_keypoints[idx:idx+3]
            if prob <= _VISIBILITY_TRESH:
                if idx/3 in _REFs.keys():
                    return False
            else:   
                c = c + 1
        if c >= _MINKPTs:
            return True
        else:
            return False

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
"""
def get_endjoints():
    #Create a dict containing all end points (e.g. RWrist, LWrist)
    part_pairs = [item for tupl in _KPTS_LINES for item in tupl] #plain list without tuples
    endjoints = dict()
    frequencies = collections.Counter(part_pairs)
    for (joint,freq) in frequencies.items():
        if freq == 1:
            endjoints.update({joint: _BODY_PART_MAPPING[joint]})
    return endjoints

def get_endjoints_depth2():
    #Create a dict for listing all lines from end points of depth 2
    endjoints_depth2 = dict()
    endjoints = get_endjoints()
    print("endjoints:",endjoints)
    for k in  endjoints.keys():
        d1 = [line for line in _KPTS_LINES if line[0] == k]
        d1.extend([(line[1],line[0]) for line in _KPTS_LINES if line[1] == k])
        
        for (end,middle) in d1:
            d2 = [line for line in _KPTS_LINES if line[0] == middle and line[1] != end]
            d2.extend([(line[1],line[0]) for line in _KPTS_LINES if line[1] == middle and line[0] != end])
            for (middle,begin) in d2:
                endjoints_depth2.update({end: begin})
    print("endjoints_depth2:",endjoints_depth2)
    exit(1)
    return endjoints_depth2
"""
def get_planes(plane_points):
    planes = []
    for p in plane_points:
        indices = list(p.keys())
        planes.append(Polygon([ keypoints[indices[0]],keypoints[indices[1]],keypoints[indices[2]] ]))
    return planes

if __name__=="__main__":
   main()





    





