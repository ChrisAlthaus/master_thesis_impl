#Creates custom features from skeleton pose annotations based on geometric properties.
import os
import json 
import argparse
import numpy as np
import math
from shapely.geometry import LineString, Point, Polygon
import scipy
import collections
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('-inputFile',required=True,
                    help='File with keypoint annotations/ predictions.')

args = parser.parse_args()

if not os.path.isfile(args.inputFile):
    raise ValueError("No valid input file.")

test_entry = {"image_id": 36, "category_id": 1, "score": 0.7828, "bbox": [167.58, 162.89, 309.61, 464.18999999999994], "keypoints": [252.122, 238.593, 1.0, 270.254, 220.46, 1.0, 233.989, 229.527, 1.0, 311.052, 229.527, 1.0, 224.923, 252.192, 1.0, 360.916, 342.854, 1.0, 215.857, 351.92, 1.0, 401.714, 483.381, 1.0, 215.857, 515.112, 1.0, 442.512, 619.374, 1.0, 247.589, 451.649, 1.0, 360.916, 587.642, 1.0, 270.254, 596.708, 1.0, 342.784, 628.44, 1.0, 261.188, 628.44, 1.0, 324.652, 623.907, 1.0, 256.655, 619.374, 1.0]}
_KEYPOINT_THRESHOLD = 0.5
_REFs = {5: "left_shoulder", 6: "right_shoulder"}
#_REFs = {1: "left_shoulder"}
_MINKPTs = 10
_NUMKPTS = 17
# In order to get the list of all files that ends with ".json"
# we will get list of all files, and take only the ones that ends with "json"
def main():
    calculateGPD(test_entry['keypoints'])
    exit(1)
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
    #json_data = [test_entry]
    for i,person in enumerate(json_data):
        if "keypoints" in person:
            #print("PREV:",person['keypoints'])
            isvalid = filterKeypoints(person['keypoints'])
            #print("LATER:",person['keypoints'])

            #if isvalid is False:
            #    exit(1)
            if isvalid:
                keypoint_descriptor = calculateGPD(person['keypoints'])
                image_id = person["image_id"]
                if image_id != prevImgId:
                    json_data.append({"image_id": image_id, "gpd": [keypoint_descriptor]})
                else:
                    json_data[-1]['gpd'].append(keypoint_descriptor)
    
    json_file = 'geometric_pose_descriptor'
    with open(os.path.join(output_dir,json_file), 'w') as f:
        print("Writing to file: ",os.path.join(output_dir,json_file))
        json.dump(json_out, f)

def filterKeypoints(pose_keypoints):
    #Filter out keypoints above a treshold
    #Skip pose if too few keypoints or reference point is not contained
    
    for idx in range(0,_NUMKPTS,3):
        x, y, prob = pose_keypoints[idx:idx+3]
        if prob <= _KEYPOINT_THRESHOLD:
            #Set visible value to indicate invalid keypoint
            pose_keypoints[idx+2] = None
            if idx/3 in _REFs.keys():
                #print("no ref key")
                return False
    if sum(x is not None for x in pose_keypoints) >= _MINKPTs:
        return True
    else:
        return False
      
     
def calculateGPD(keypoints):
    xs = keypoints[::3]
    ys = keypoints[1::3]
    vs = keypoints[2::3]

    num_kps = 17
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
    elif len(_REFs) == 1:
        pmid = [xs[ref_ids[0]], ys[ref_ids[0]]]
    else:
        raise ValueError("_REFS not valid.")

    ref_point = np.array(pmid)
    keypoints = np.asarray(list(zip(xs,ys)), np.float32)
    
    keypoints = keypoints - ref_point
 
    pose_descriptor = []
    #Joint Coordinate: all keypoints in relative coordinate system + reference point in world coordinate system
    #Dimension 25 x 2 + 1 = 51 
    #joint_coordinates = np.delete(keypoints,inv_body_part_mapping['MidHip'])
    joint_coordinates = np.append(ref_point,keypoints)
    print("Dimension joint coordinates: ", joint_coordinates.shape) 
    pose_descriptor.append(joint_coordinates.tolist())

    indices_pairs = []
    JJ_d = joint_joint_distances(keypoints,indices_pairs=None)
    pose_descriptor.append(JJ_d)
    
    indices_pairs = []
    JJ_o = joint_joint_orientations(keypoints, indices_pairs=None)
    pose_descriptor.append(JJ_o)

    l_adjacent = lines_direct_adjacent(keypoints, kpts_lines)

    l_end_depth2 = lines_endjoints_depth2(keypoints, end_joints_depth2)

    indices_pairs = []
    line_endjoints = lines_endjoints(keypoints, end_joints, indices_pairs = None)

    l_custom = lines_custom(keypoints)
    #Merge all lines
    l_adjacent.update(l_end_depth2)
    l_adjacent.update(line_endjoints)
    l_adjacent.update(l_custom)


    kpt_line_mapping = {7:[(5,9),'left_arm'], 8:[(6,10),'right_arm'], 13:[(11,15),'left_leg'], 14:[(12,16),'right_leg'], 
                        3:[(5,0),'shoulder_head_left'], 4:[(6,0),'shoulder_head_right'],
                        6:[(8,5),'shoulders_elbowr'], 5:[(6,7),'shoulders_elbowsl'], 
                        13:[(14,15),'knees_foot_side'], 14:[(13,16),'knees_foot_side'], 
                        10:[(5,9),'arms_left_side'], 9:[(6,10),'arms_right_side'],
                        0:[(16,12),'headpos_side'], 0:[(15,11),'headpos_side']} 
    JL_d = joint_line_distances(keypoints, l_adjacent, kpt_line_mapping)
    pose_descriptor.append(JL_d)

    line_line_mapping = []
    LL_a = line_line_angles(l_adjacent)
    pose_descriptor.append(LL_a)

    #Planes for selected regions: 2 for each head, arms, leg & foot region 
    #plane_points = [{1: 'Neck',0: 'Nose', 18: 'LEar'}, {1: 'Neck',0: 'Nose', 18: 'REar'}, {2: 'RShoulder', 3: 'RElbow', 4: 'RWrist'},
    #                {5: 'LShoulder', 6: 'LElbow',7: 'LWrist'}, {12: 'LHip', 13: 'LKnee', 14: 'LAnkle'}, { 9: 'RHip', 10: 'RKnee', 11: 'RAnkle'},
    #                {13: 'LKnee', 14: 'LAnkle', 19: 'LBigToe'}, {10: 'RKnee', 11: 'RAnkle', 22: 'RBigToe'}]

    #planes = get_planes(plane_points)
    #JP_d = joint_plane_distances(keypoints, planes)


 
    #print(pose_descriptor)
    print("Dimension of pose descriptor: ", len(pose_descriptor)) 
    print("\n")

    return pose_descriptor

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
    
    print("Dimension joint distances: ", len(joint_distances)) 
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
    print("Dimension of joint orientations: ",len(joint_orientations))
    return joint_orientations

def lines_direct_adjacent(keypoints, kpts_lines):
    lines = {}
    #Directly adjacent lines, Dimesions: 25 - 1 / 17-1 = 16
    #Lines pointing outwards
    for start,end in kpts_lines:
        lines.update({(start,end) : LineString([keypoints[start], keypoints[end]])})
    print("Dimension lines between directly adjacent keypoints: ", len(lines))
    return lines

def lines_endjoints_depth2(keypoints, end_joints_depth2):
    lines = {}
    #Kinetic chain from end joints of depth 2
    #Dimension (BODY_MODEL_25): 8 / 6
    for (end, begin) in end_joints_depth2.items():
        lines.update({(begin,end) : LineString([keypoints[begin], keypoints[end]])})
    print("Dimension lines from end-joints of depth 2: ", len(lines))
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
    print("Dimension lines between end-joints only: ", len(lines)) 
    return lines

def lines_custom(keypoints):
    kpt_mapping = {14:15, 13:16, #foot knees x
                    5:0,  0:6,  #shoulders-nose
                    8:5, 6:7, #elbows-shoulders x
                    }
    lines = {}
    for k1,k2 in kpt_mapping.items():
        lines.update({(k1,k2) : LineString([keypoints[k1], keypoints[k2]])})
    return lines

def joint_line_distances(keypoints, lines, kpt_line_mapping = None):
    #Joint-Line Distance
    
    joint_line_distances = []
    if kpt_line_mapping is None:
        #Approx. Dimension: 60 lines * (25-3) joints = 1320 / (16+6+15) lines * (17-3) joints = 518
        for l in lines:
            coords = list(l.coords)
            for joint in keypoints:
                if tuple(joint) not in coords:
                    joint_line_distances.append(Point(joint).distance(l))
                #To provide static descriptor dimension
                else:
                    joint_line_distances.append(0)
    else:
        for k, [(k1,k2),label] in kpt_line_mapping.items():
            if (k1,k2) in lines.keys():
                joint_line_distances.append(Point(keypoints[k]).distance(lines[(k1,k2)]))
            elif (k2,k1) in lines.keys():
                joint_line_distances.append(Point(keypoints[k]).distance(lines[(k2,k1)]))
            else:
                print("Not found line:" ,k1,k2)
                #raise ValueError("Line in keypoint-line mapping not in line storage.")
    
    print("Dimension joint line distances: ", len(joint_line_distances)) 
    return joint_line_distances

def line_line_angles(lines, line_line_mapping = None):
    #Line-Line Angle
    line_line_angles = []    
    def angle(l1,l2):
        [p11,p12] = list(l1.coords)
        [p21,p22] = list(l2.coords)
        #limb vectors pointing outwards
        j1 = np.subtract(p12,p11)
        j2 = np.subtract(p22,p21)
        return np.rad2deg(np.arccos(np.dot(j1, j2) / (np.linalg.norm(j1) * np.linalg.norm(j2))))
    if line_line_mapping is None:
        #(25-1) over 2 = 276 / (17-1) over 2 = 120
        finished = []
        for (k11,k12),l1 in lines.items():
            for (k21,k22),l2 in lines.items():
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
        for (k11,k12),(k21,k22) in line_line_mapping.items():
            l1 = lines[(k11,k12)]
            l2 = lines[(k21,k22)]
            if l1!=l2:
                line_line_angles.append(angle(l1,l2))
    print("Dimension line line angles: ", len(line_line_angles)) 
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
    print("Dimension joint plane distances: ", len(joint_plane_distances)) 
    return joint_plane_distances

if __name__=="__main__":
   main()





    





