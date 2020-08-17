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

{"image_id": 36, "category_id": 1, "score": 0.7828, "bbox": [167.58, 162.89, 309.61, 464.18999999999994], "keypoints": [252.122, 238.593, 1.0, 270.254, 220.46, 1.0, 233.989, 229.527, 1.0, 311.052, 229.527, 1.0, 224.923, 252.192, 1.0, 360.916, 342.854, 1.0, 215.857, 351.92, 1.0, 401.714, 483.381, 1.0, 215.857, 515.112, 1.0, 442.512, 619.374, 1.0, 247.589, 451.649, 1.0, 360.916, 587.642, 1.0, 270.254, 596.708, 1.0, 342.784, 628.44, 1.0, 261.188, 628.44, 1.0, 324.652, 623.907, 1.0, 256.655, 619.374, 1.0]}

# In order to get the list of all files that ends with ".json"
# we will get list of all files, and take only the ones that ends with "json"
def main():

    output_dir = os.path.join('/home/althausc/master_thesis_impl/posedescriptors/out', datetime.datetime.now().strftime('%m/%d_%H-%M-%S'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        raise ValueError("Output directory %s already exists."%output_dir)
   
    print("Reading from file: ",args.inputFile)
    with open (args.inputFile, "r") as f:
        json_data = json.load(f)
    json_out = []
    for i,person in enumerate(json_data):
        if "keypoints" in person:
            keypoints_x = person["keypoints"][::3]
            keypoints_y = person["keypoints"][1::3]
            keypoint_descriptor = calculateGPD(keypoints_x,keypoints_y)
            image_id = person["image_id"]

            json_out.append({"image_id": image_id, "gpd": keypoint_descriptor})
            exit(1)
    
    json_file = 'geometric_pose_descriptor'
    with open(os.path.join(output_dir,json_file), 'w') as f:
        print("Writing to file: ",os.path.join(output_dir,json_file))
        json.dump(json_out, f)


def calculateGPD(xs,ys):
    body_part_mapping = {0: 'Nose', 1: 'Neck', 2: 'RShoulder', 3: 'RElbow', 4: 'RWrist', 5: 'LShoulder', 6: 'LElbow',
                         7: 'LWrist', 8: 'MidHip', 9: 'RHip', 10: 'RKnee', 11: 'RAnkle', 12: 'LHip', 13: 'LKnee', 14: 'LAnkle',
                         15: 'REye', 16: 'LEye', 17: 'REar', 18: 'LEar', 19: 'LBigToe', 20: 'LSmallToe', 21: 'LHeel', 
                         22: 'RBigToe', 23: 'RSmallToe', 24: 'RHeel', 25: 'Background'}
    # Get also with print(op.getPosePartPairs(poseModel))
    part_pairs = [1, 8, 1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 10, 11, 8, 12, 12, 13, 13, 14, 1, 0, 0, 15,
                  15, 17, 0, 16, 16, 18, 2, 17, 5, 18, 14, 19, 19, 20, 14, 21, 11, 22, 22, 23, 11, 24]

        COCO_PERSON_KEYPOINT_NAMES = (
        0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear", 5: "left_shoulder", 6: "right_shoulder",
        7: "left_elbow", 8: "right_elbow", 9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
        13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle",
    )
    # fmt: on   #TODO

    # Pairs of keypoints that should be exchanged under horizontal flipping
    COCO_PERSON_KEYPOINT_FLIP_MAP = (
        ("left_eye", "right_eye"),
        ("left_ear", "right_ear"),
        ("left_shoulder", "right_shoulder"),
        ("left_elbow", "right_elbow"),
        ("left_wrist", "right_wrist"),
        ("left_hip", "right_hip"),
        ("left_knee", "right_knee"),
        ("left_ankle", "right_ankle"),
    )

    # rules for pairs of keypoints to draw a line between, and the line color to use.
    KEYPOINT_CONNECTION_RULES = [
        # face
        ("left_ear", "left_eye", (102, 204, 255)),
        ("right_ear", "right_eye", (51, 153, 255)),
        ("left_eye", "nose", (102, 0, 204)),
        ("nose", "right_eye", (51, 102, 255)),
        # upper-body
        ("left_shoulder", "right_shoulder", (255, 128, 0)),
        ("left_shoulder", "left_elbow", (153, 255, 204)),
        ("right_shoulder", "right_elbow", (128, 229, 255)),
        ("left_elbow", "left_wrist", (153, 255, 153)),
        ("right_elbow", "right_wrist", (102, 255, 224)),
        # lower-body
        ("left_hip", "right_hip", (255, 102, 0)),
        ("left_hip", "left_knee", (255, 255, 77)),
        ("right_hip", "right_knee", (153, 255, 204)),
        ("left_knee", "left_ankle", (191, 255, 128)),
        ("right_knee", "right_ankle", (255, 195, 77)),
    ]
    #Create a dict containing all end points (e.g. RWrist, LWrist)
    end_joints = dict()
    frequencies = collections.Counter(part_pairs)
    for (joint,freq) in frequencies.items():
        if freq == 1:
            end_joints.update({joint: body_part_mapping[joint]})
    #Create a dict for listing all lines from end points of depth 2
    end_joints_depth2 = dict()
    part_pair_points = zip(part_pairs[0::2], part_pairs[1::2])
    for k in end_joints.keys():
        d1 = [line for line in part_pair_points if line[0] == k]
        d1.extend([(line[1],line[0]) for line in part_pair_points if line[1] == k])

        for (end,middle) in d1:
            d2 = [line for line in part_pair_points if line[0] == middle and line[1] != end]
            d2.extend([(line[1],line[0]) for line in part_pair_points if line[1] == middle and line[0] != end])
            for (middle,begin) in d2:
                end_joints_depth2.update({end: begin})

    #end_joints = {4: 'RWrist', 7: 'LWrist', 17: 'REar', 18: 'LEar', 20: 'LSmallToe', 21: 'LHeel', 23: 'RSmallToe', 24: 'RHeel'}
    #end_joints_depth2 = {4: 2, 7: 5, 17: 'REar', 18: 'LEar', 20: 'LSmallToe', 21: 'LHeel', 23: 'RSmallToe', 24: 'RHeel'}

    inv_body_part_mapping = dict((v,k) for k, v in body_part_mapping.items())

    #Delete background label if present & included in keypoint list 
    if len(xs) == len(body_part_mapping) and len(ys) == len(body_part_mapping):
        if 'Background' in inv_body_part_mapping:
            del xs[inv_body_part_mapping['Background']]
            del ys[inv_body_part_mapping['Background']]

    # Make all keypoints relative to a selected reference point
    ref_point = (xs[inv_body_part_mapping['MidHip']], ys[inv_body_part_mapping['MidHip']])
    keypoints = np.asarray(list(zip(xs,ys)), np.float32)
 
    keypoints = keypoints - ref_point

    pose_descriptor = []
    #Joint Coordinate: all keypoints in relative coordinate system + reference point in world coordinate system
    #Dimension 25 x 2 + 1 = 51 
    joint_coordinates = np.delete(keypoints,inv_body_part_mapping['MidHip'])
    joint_coordinates = np.append(joint_coordinates, ref_point)
    print("Dimension joint coordinates: ", joint_coordinates.shape) 
    pose_descriptor.append(joint_coordinates.tolist())

    #Joint-Joint Distance
    #Dimension 25 over 2 = 300
    #scipy.special.binom(len(keypoints), 2))
    joint_distances = []
    #Only consider unique different joint pairs ,e.g. (1,2) == (2,1)
    for i1 in range(len(keypoints)):
        for i2 in range(len(keypoints)):
            if i1 < i2:
                joint_distances.append(np.linalg.norm(keypoints[i2]-keypoints[i1]))
    print("Dimension joint distances: ", len(joint_distances)) 
    pose_descriptor.append(joint_distances)

    #Joint-Joint Orientation (vector orientation of unit vector from j1 to j2)
    #Dimension 25 over 2 = 300
    joint_orientations = []
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
    print("Dimension joint orientations: ", len(joint_orientations)) 
    pose_descriptor.append(joint_orientations)

    lines = []
    lines_adjacent_only = []
    #Directly adjacent lines, Dimesions: 25 - 1
    #Lines pointing outwards
    for i in range(len(keypoints)-1):
        lines.append(LineString([keypoints[i], keypoints[i+1]]))
        lines_adjacent_only.append(LineString([keypoints[i], keypoints[i+1]]))
    #Kinetic chain from end joints of depth 2
    #Dimension (BODY_MODEL_25): 8
    for (end, begin) in end_joints_depth2.items():
        lines.append(LineString([keypoints[begin], keypoints[end]]))
    #Lines only between end joints
    #Dimensions: num(end_joints) over 2 , BODY_MODEL_25: 8 over 2 = 28
    for (k1,label1) in end_joints.items():
        for (k2,label2) in end_joints.items():
            if k1!=k2:
                lines.append(LineString([keypoints[k1], keypoints[k2]]))

    #Joint-Line Distance
    #Approx. Dimension: 60 lines * (25-3) joints = 1320
    joint_line_distances = []
    for l in lines:
        coords = list(l.coords)
        for joint in keypoints:
            if tuple(joint) not in coords:
                joint_line_distances.append(Point(joint).distance(l))
            #To provide static descriptor dimension
            else:
                joint_line_distances.append(0)
    print("Dimension joint line distances: ", len(joint_line_distances)) 
    pose_descriptor.append(joint_line_distances) 

    #Line-Line Angle
    #(25-1) over 2 = 276
    line_line_angles = []
    for i1 in range(len(lines_adjacent_only)):
        for i2 in range(len(lines_adjacent_only)):
            if i1<i2:
                l1 = lines_adjacent_only[i1]
                l2 = lines_adjacent_only[i2]

                if l1!=l2:
                    [p11,p12] = list(l1.coords)
                    [p21,p22] = list(l2.coords)
                    #limb vectors pointing outwards

                    j1 = np.subtract(p12,p11)
                    j2 = np.subtract(p22,p21)
                    #don't consider origin(zero-point)
                    if np.any(j1) and np.any(j2):  
                        angle = np.rad2deg(np.arccos(np.dot(j1, j2) / (np.linalg.norm(j1) * np.linalg.norm(j2))))
                        line_line_angles.append(angle)
                    else:
                        line_line_angles.append(0)
                #To provide static descriptor dimension
                else:
                    line_line_angles.append(0)
    print("Dimension line line angles: ", len(line_line_angles)) 
    pose_descriptor.append(line_line_angles)

    #Planes for selected regions: 2 for each head, arms, leg & foot region 
    plane_points = [{1: 'Neck',0: 'Nose', 18: 'LEar'}, {1: 'Neck',0: 'Nose', 18: 'REar'}, {2: 'RShoulder', 3: 'RElbow', 4: 'RWrist'},
                    {5: 'LShoulder', 6: 'LElbow',7: 'LWrist'}, {12: 'LHip', 13: 'LKnee', 14: 'LAnkle'}, { 9: 'RHip', 10: 'RKnee', 11: 'RAnkle'},
                    {13: 'LKnee', 14: 'LAnkle', 19: 'LBigToe'}, {10: 'RKnee', 11: 'RAnkle', 22: 'RBigToe'}]
    planes = []
    for p in plane_points:
        indices = list(p.keys())
        planes.append(Polygon([ keypoints[indices[0]],keypoints[indices[1]],keypoints[indices[2]] ]))
    
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
    pose_descriptor.append(joint_plane_distances)
 
    #print(pose_descriptor)
    print("Dimension of pose descriptor: ", len(pose_descriptor)) 
    print("\n")

    return pose_descriptor
   

if __name__=="__main__":
   main()





    





