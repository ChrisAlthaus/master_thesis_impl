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

parser = argparse.ArgumentParser()
parser.add_argument('-inputFile',required=True,
                    help='File with keypoint annotations/ predictions.')
parser.add_argument("-mode", type=int, help="Specify types of features which will be computed.")
parser.add_argument("-pca", type=int, help="Specify dimensions of pca vector.")
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")
args = parser.parse_args()

test_entry = {'image_id': 23899057496, 'category_id': 1, 'bbox': [211.99081420898438, 139.43743896484375, 425.96087646484375, 355.24871826171875], 'keypoints': [334.2212219238281, 201.67015075683594, 1.079627275466919, 331.54656982421875, 189.38385009765625, 1.7378227710723877, 312.2892761230469, 192.58897399902344, 1.028214931488037, 334.7561340332031, 202.20433044433594, 0.08344336599111557, 269.4952697753906, 213.9564208984375, 0.38487914204597473, 346.5245056152344, 262.033203125, 0.13131119310855865, 288.2176513671875, 285.5373840332031, 0.10808556526899338, 425.1584777832031, 354.4474182128906, 0.020250316709280014, 383.434326171875, 328.8064880371094, 0.012223891913890839, 276.44927978515625, 354.4474182128906, 0.01989334262907505, 425.1584777832031, 354.4474182128906, 0.020259613171219826, 425.1584777832031, 354.4474182128906, 0.02405051700770855, 403.761474609375, 354.4474182128906, 0.02277219668030739, 425.1584777832031, 354.4474182128906, 0.03073735162615776, 425.1584777832031, 354.4474182128906, 0.03939764201641083, 425.1584777832031, 354.4474182128906, 0.02348250150680542, 425.1584777832031, 354.4474182128906, 0.03718782961368561], 'score': 0.9582511186599731}
_KEYPOINT_THRESHOLD = 0.5
_REFs = {5: "left_shoulder", 6: "right_shoulder"}
#_REFs = {1: "left_shoulder"}
_MINKPTs = 10
_NUMKPTS = 17
_MODES = ['JcJLdLLa_reduced', 'JLd_all']
_FILTER = 1
_FILTERMODE = 1 #['strict' vs 'nostrict'] 

if not os.path.isfile(args.inputFile):
    raise ValueError("No valid input file.")
if args.verbose:
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')
if args.mode not in range(0, len(_MODES)):
     raise ValueError("No valid mode number.")


def main():
    #calculateGPD(test_entry['keypoints'])
    #exit(1)
    output_dir = os.path.join('/home/althausc/master_thesis_impl/posedescriptors/out', datetime.datetime.now().strftime('%m/%d_%H-%M-%S'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        raise ValueError("Output directory %s already exists."%output_dir)
    
    print("Reading from file: ",args.inputFile)
    with open (args.inputFile, "r") as f:
        json_data = json.load(f)

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
            isvalid = filterKeypoints(person['keypoints'], _FILTERMODE)
            #logging.debug("LATER:",person['keypoints'])
            
            if isvalid:
                keypoint_descriptor, visibilities = calculateGPD(person['keypoints'], _MODES[args.mode])
                json_out.append({"image_id": person["image_id"], "gpd": keypoint_descriptor, 'score': person['score'], 'vis': visibilities})
                c = c + 1
        if i%1000 == 0 and i!=0:
            print("Processed %d elements."%i)

    print("Time for calculating %d descriptors = %s seconds " % (c,time.time() - start_time))
    print("Original number of json predictions: ",len(json_data))
    print("Number of calculated descriptors: ",c)

    if args.pca is not None:
        model = applyPCA(json_out, args.pca)
        pickle.dump(model, open(os.path.join(output_dir,'modelpca%d'%args.pca + '.pkl'), "wb"))

    json_file = 'geometric_pose_descriptor_c_%d_m%d_t%.2f_f%d.%d_mkpt%d'%(c,args.mode, _KEYPOINT_THRESHOLD, _FILTER, _FILTERMODE, _MINKPTs)
    with open(os.path.join(output_dir, json_file+'.json'), 'w') as f:
        print("Writing to file: ",os.path.join(output_dir,json_file+'.json'))
        json.dump(json_out, f)

def applyPCA(json_data, dim):
    gpds = [item['gpd'] for item in json_data]
    pca = PCA(n_components=dim)
    pca_result = pca.fit_transform(gpds)

    for i,item in enumerate(json_data):
        item['gpd'] = list(pca_result[i])
    
    return pca
       

def filterKeypoints(pose_keypoints, strict=False):
    #Filter out keypoints above a treshold
    #Skip pose if too few keypoints or reference point is not contained
    
    if not strict:
    #Write None to keypoints for each vis under the threshold
    #For later descriptors None values will be replaced by default value (=0)
        for idx in range(0,_NUMKPTS):
            x, y, prob = pose_keypoints[idx:idx+3]
            if prob <= _KEYPOINT_THRESHOLD:
                #Set visible value to indicate invalid keypoint
                pose_keypoints[idx+2] = None
                if idx/3 in _REFs.keys():
                    #if reference point is unstable, skip this pose
                    #logging.debug("no ref key")
                    return False
        if sum(x is not None for x in pose_keypoints) >= _MINKPTs:
            return True
        else:
            return False
    else:
    #Keypoints are sorted out or given back fully/not reduced
        c = 0
        for idx in range(0,_NUMKPTS):
            x, y, prob = pose_keypoints[idx:idx+3]
            if prob <= _KEYPOINT_THRESHOLD:
                if idx/3 in _REFs.keys():
                    return False
            else:   
                c = c + 1
        if c >= _MINKPTs:
            return True
        else:
            return False
      
     
def calculateGPD(keypoints, mode):
    xs = keypoints[::3]
    ys = keypoints[1::3]
    vs = keypoints[2::3]

    num_kps = 17
    kpts_valid = [1 if v is not None else 0 for v in vs] #Keypoints with visibility None should not be considered (0 in feature decsriptor)
    logging.debug("kpts_valid: {}".format(kpts_valid))

    body_part_mapping = {
        0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear", 5: "left_shoulder", 6: "right_shoulder",
        7: "left_elbow", 8: "right_elbow", 9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
        13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"}
       
    kpts_lines = [(1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (5, 6), (11, 12)]
    part_pairs = [item for tupl in kpts_lines for item in tupl] #plain list without tuples
    
    # Pairs of keypoints that should be exchanged under horizontal flipping
    #kps_symmetry = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

    end_joints = get_endjoints(part_pairs,body_part_mapping)
    end_joints_depth2 = get_endjoints_depth2(end_joints, kpts_lines)
  
    #end_joints = {4: 'RWrist', 7: 'LWrist', 17: 'REar', 18: 'LEar', 20: 'LSmallToe', 21: 'LHeel', 23: 'RSmallToe', 24: 'RHeel'}
    #end_joints_depth2 = {4: 0, 3: 0, 10: 6, 9: 5, 16: 12, 15: 11}

    inv_body_part_mapping = dict((v,k) for k, v in body_part_mapping.items())

    #Delete background label if present & included in keypoint list 
    if len(xs) == len(body_part_mapping) and len(ys) == len(body_part_mapping):
        if 'Background' in inv_body_part_mapping:
            del xs[inv_body_part_mapping['Background']]
            del ys[inv_body_part_mapping['Background']]
   
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
    #logging.debug("ref point: ",ref_point)
    keypoints = np.asarray(list(zip(xs,ys)), np.float32)
    keypoints = (keypoints - ref_point).tolist()

    pose_descriptor = []
    
    if mode == 'JcJLdLLa_reduced':
        #Dimensions: 18 keypoints (with or without visibility flag)
        joint_coordinates = joint_coordinates_rel(keypoints, ref_point.tolist())
        #joint_coordinates = joint_coordinates_rel(keypoints, ref_point.tolist(), visiblities = vs , vclipping = True)

        pose_descriptor.append(joint_coordinates)

        indices_pairs = []
        #JJ_d = joint_joint_distances(keypoints,indices_pairs=None)
        #pose_descriptor.append(JJ_d)
        
        indices_pairs = []
        #JJ_o = joint_joint_orientations(keypoints, indices_pairs=None)
        #pose_descriptor.append(JJ_o)

        l_adjacent = lines_direct_adjacent(keypoints, kpts_lines)
        l_end_depth2 = lines_endjoints_depth2(keypoints, end_joints_depth2)

        indices_pairs = []
        line_endjoints = lines_endjoints(keypoints, end_joints, indices_pairs = None)

        l_custom = lines_custom(keypoints)
        #Merge all lines
        l_adjacent.update(l_end_depth2)
        l_adjacent.update(line_endjoints)
        l_adjacent.update(l_custom)

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
        JL_d = joint_line_distances(keypoints, l_adjacent, kpts_valid, kpt_line_mapping)
        pose_descriptor.append(JL_d)

        #Dimensions: 10 angles
        line_line_mapping = {(10,9):[(9,15),'hands_lfoot'], (9,10):[(9,16),'hands_rfoot'],
                            (10,16):[(9,10),'hands_lfoot'], (16,10):[(10,15),'hands_rfoot'],
                            (5,11):[(5,9),'hand_shoulder_hip_l'], (6,12):[(6,10),'hand_shoulder_hip_r'],
                            (6,8):[(5,7),'upper_arms'], (8,10):[(7,9),'lower_arms'],
                            (12,14):[(11,13),'upper_legs'], (14,16):[(13,15),'lower_legs'],
                            (0,5):[(3,5),'head_shoulder_l'], (4,6):[(0,6),'head_shoulder_r']}
        LL_a = line_line_angles(l_adjacent, kpts_valid, line_line_mapping)
        pose_descriptor.append(LL_a)

    elif mode == 'JLd_all':
        l_adjacent = lines_direct_adjacent(keypoints, kpts_lines)
        JL_d = joint_line_distances(keypoints, l_adjacent, kpts_valid)
        pose_descriptor.append(JL_d)

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
    vs = list(map(lambda x: 0 if x is None else x, vs))
    visibilities = list(map(lambda y: max(0,min(2,y)), vs))


    return pose_descriptor, visibilities

def get_endjoints(part_pairs, body_part_mapping):
    #Create a dict containing all end points (e.g. RWrist, LWrist)
    end_joints = dict()
    frequencies = collections.Counter(part_pairs)
    for (joint,freq) in frequencies.items():
        if freq == 1:
            end_joints.update({joint: body_part_mapping[joint]})
    return end_joints

def get_endjoints_depth2(end_joints, kpts_lines):
    #Create a dict for listing all lines from end points of depth 2
    end_joints_depth2 = dict()
    
    for k in end_joints.keys():
        d1 = [line for line in kpts_lines if line[0] == k]
        d1.extend([(line[1],line[0]) for line in kpts_lines if line[1] == k])
        
        for (end,middle) in d1:
            d2 = [line for line in kpts_lines if line[0] == middle and line[1] != end]
            d2.extend([(line[1],line[0]) for line in kpts_lines if line[1] == middle and line[0] != end])
            for (middle,begin) in d2:
                end_joints_depth2.update({end: begin})
    return end_joints_depth2

def joint_coordinates_rel(keypoints, reference_point, visiblities = None, vclipping = False):
    #Joint Coordinate: all keypoints in relative coordinate system + reference point in world coordinate system
    joint_coordinates = []
    
    if visiblities is None:
        #Dimension 17 x 2 + 1 x 2 = 36 
        #joint_coordinates = np.delete(keypoints,inv_body_part_mapping['MidHip'])
        joint_coordinates = itertools.chain([reference_point], keypoints)
        joint_coordinates = [item for sublist in list(joint_coordinates) for item in sublist]
        logging.debug("Dimension joint coordinates: {}".format(len(joint_coordinates))) 
        return joint_coordinates

    else:
        #Adding the visibility flag
        #Meaning:   visibility == 0 that keypoint not in the image
        #           visibility == 1 that keypoint is in the image BUT not visible, 
        #           visibility == 2 that keypoint is visible
        #Dimension 17 x 3 + 1 x 3 = 54 
        joint_coordinates = itertools.chain([reference_point], keypoints)
        if vclipping:
            #Clip values bcose sometime very large
            visiblities = map(lambda y: max(0,min(2,y)), visiblities)
        joint_coordinates = [(xy[0],xy[1],v) for [xy, v] in zip(joint_coordinates, visiblities)]
        joint_coordinates = [item for sublist in joint_coordinates for item in sublist]
        logging.debug("Dimension joint coordinates: {}".format(len(joint_coordinates))) 
        return joint_coordinates
    

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
 
    return joint_distances  

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
    return joint_orientations

def lines_direct_adjacent(keypoints, kpts_lines):
    lines = {}
    #Directly adjacent lines, Dimesions: 25 - 1 / 17-1 = 16
    #Lines pointing outwards
    for start,end in kpts_lines:
        lines.update({(start,end) : LineString([keypoints[start], keypoints[end]])})
    logging.debug("\tDimension lines between directly adjacent keypoints: {}".format(len(lines)))
    return lines

def lines_endjoints_depth2(keypoints, end_joints_depth2):
    lines = {}
    #Kinetic chain from end joints of depth 2
    #Dimension (BODY_MODEL_25): 8 / 6
    for (end, begin) in end_joints_depth2.items():
        lines.update({(begin,end) : LineString([keypoints[begin], keypoints[end]])})
    logging.debug("\tDimension lines from end-joints of depth 2: {}".format(len(lines)))
    return lines

def lines_endjoints(keypoints, end_joints, indices_pairs = None):
    lines = {}
    #Lines only between end joints
    if indices_pairs is None:
        #Dimensions: num(end_joints) over 2 , BODY_MODEL_25: 8 over 2 = 28 / 6 over 2 = 15
        for (k1,label1) in end_joints.items():
            for (k2,label2) in end_joints.items():
                if k1!=k2:
                    lines.update({(k1,k2) : LineString([keypoints[k1], keypoints[k2]])})
    else:
        for start,end in indices_pairs:
            if start in end_joints.keys() and end in end_joints.keys():
                lines.update({(start,end) : LineString([keypoints[start], keypoints[end]])})
            else:
                raise ValueError("Given index is no end joint.")
    logging.debug("\tDimension lines between end-joints only: {}".format(len(lines))) 
    return lines

def lines_custom(keypoints):
    kpt_mapping = {14:15, 13:16, #foot knees x
                    5:0,  0:6,  #shoulders-nose
                    8:5, 7:6, #elbows-shoulders x
                    11:5, 6:12, #shoulders-hips
                    3:5, 4:6  #shoulders-ears
                    }
    lines = {}
    for k1,k2 in kpt_mapping.items():
        lines.update({(k1,k2) : LineString([keypoints[k1], keypoints[k2]])})
    logging.debug("\tDimension lines custom: {}".format(len(lines))) 
    return lines

def joint_line_distances(keypoints, lines, kpts_valid, kpt_line_mapping = None):
    #Joint-Line Distance
    
    joint_line_distances = []
    
    if kpt_line_mapping is None:
        #Approx. Dimension: 60 lines * (25-3) joints = 1320 / (16+6+15) lines * (17-3) joints = 518
        for k, l in lines.items():
            coords = list(l.coords)
            for i,joint in enumerate(keypoints):
                if not kpts_valid[i]:
                    joint_line_distances.append(0)
                    continue
                if tuple(joint) not in coords:
                    joint_line_distances.append(Point(joint).distance(l))
                #To provide static descriptor dimension
                else:
                    joint_line_distances.append(0)
    else:
        for k, item in kpt_line_mapping.items():
            if isinstance(item[0], list): 
                for [(k1,k2),label] in item:
                    if not kpts_valid[k1] or not kpts_valid[k2] or not kpts_valid[k]:
                        joint_line_distances.append(0)
                        continue
                    if (k1,k2) in lines.keys():
                        joint_line_distances.append(Point(keypoints[k]).distance(lines[(k1,k2)]))
                    elif (k2,k1) in lines.keys():
                        joint_line_distances.append(Point(keypoints[k]).distance(lines[(k2,k1)]))
                    else:
                        logging.debug("Not found line: {}{}".format(k1,k2))
                        #raise ValueError("Line in keypoint-line mapping not in line storage.")
            else:
                [(k1,k2),label] = item
                if not kpts_valid[k1] or not kpts_valid[k2]:
                    joint_line_distances.append(0)
                    continue                
                if (k1,k2) in lines.keys():
                    joint_line_distances.append(Point(keypoints[k]).distance(lines[(k1,k2)]))
                elif (k2,k1) in lines.keys():
                    joint_line_distances.append(Point(keypoints[k]).distance(lines[(k2,k1)]))
                else:
                    logging.debug("Not found line: {}{}".format(k1,k2))
   
    logging.debug("Dimension joint line distances: {}".format(len(joint_line_distances))) 
    return joint_line_distances

def line_line_angles(lines, kpts_valid, line_line_mapping = None):
    #Line-Line Angle
    line_line_angles = []    
    def angle(l1,l2):
        [p11,p12] = list(l1.coords)
        [p21,p22] = list(l2.coords)
        #When line is a point (due to prediction overlay kpts) return 0
        if (p11[0] == p12[0] and p11[1] == p12[1]) or (p21[0] == p22[0] and p21[1] == p22[1]):
            return 0
        #limb vectors pointing outwards
        j1 = np.subtract(p12,p11)
        j2 = np.subtract(p22,p21)

        j1_norm = j1/np.linalg.norm(j1)
        j2_norm = j2/np.linalg.norm(j2)

        return np.arccos(np.clip(np.dot(j1_norm, j2_norm), -1.0, 1.0))
 
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
        for (k11,k12),[(k21,k22),label] in line_line_mapping.items():
            if not kpts_valid[k11] or not kpts_valid[k12] or not kpts_valid[k21] or not kpts_valid[k22]:
                line_line_angles.append(0)
                continue
            l1 = lines[(k11,k12)] if (k11,k12) in lines.keys() else lines[(k12,k11)]
            l2 = lines[(k21,k22)] if (k21,k22) in lines.keys() else lines[(k22,k21)]
            if l1!=l2:
                line_line_angles.append(angle(l1,l2))
            else:
                #lines having the exact same directions
                line_line_angles.append(0)

    logging.debug("Dimension line line angles: {}".format(len(line_line_angles)) )
    return line_line_angles

def get_planes(plane_points):
    planes = []
    for p in plane_points:
        indices = list(p.keys())
        planes.append(Polygon([ keypoints[indices[0]],keypoints[indices[1]],keypoints[indices[2]] ]))
    return planes

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
    return joint_plane_distances

if __name__=="__main__":
   main()





    





