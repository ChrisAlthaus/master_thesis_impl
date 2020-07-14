#Creates custom features from skeleton pose annotations based on geometric properties.
import os
import json 
import argparse
import numpy as np
from shapely.geometry import LineString, Point, Polygon
import scipy
import collections

parser = argparse.ArgumentParser()
parser.add_argument('-inputDirectory',required=True,
                    help='Directory of OpenPose .json pose files.')
parser.add_argument('-outputDirectory',required=True,
                    help='Directory to save the computed features.')

args = parser.parse_args()

if not os.path.isdir(args.inputDirectory):
    raise ValueError("No valid input directory.")
if not os.path.isdir(args.outputDirectory):
    raise ValueError("No valid output directory.")

# In order to get the list of all files that ends with ".json"
# we will get list of all files, and take only the ones that ends with "json"

json_files = [ x for x in os.listdir(args.inputDirectory) if x.endswith("json") ]

for json_file in json_files:
    json_file_path = os.path.join(args.inputDirectory, json_file)
    json_data = None
    with open (json_file_path, "r") as f:
        json_data = json.load(f)

    json_out = dict()
    for i,person in enumerate(json_data["people"]):
        keypoints_x = person["pose_keypoints_2d"][::3]
        keypoints_y = person["pose_keypoints_2d"][1::3]

        keypoint_descriptor = calculateGPD(keypoints_x,keypoints_y)
        json_out.update({i: keypoint_descriptor})
    
    with open(os.path.join(args.outputDirectory,json_file), 'w') as f:
        json.dump(json_out, f)


def calculateGPD(xs,ys):
    body_part_mapping = {0: 'Nose', 1: 'Neck', 2: 'RShoulder', 3: 'RElbow', 4: 'RWrist', 5: 'LShoulder', 6: 'LElbow',
                         7: 'LWrist', 8: 'MidHip', 9: 'RHip', 10: 'RKnee', 11: 'RAnkle', 12: 'LHip', 13: 'LKnee', 14: 'LAnkle',
                         15: 'REye', 16: 'LEye', 17: 'REar', 18: 'LEar', 19: 'LBigToe', 20: 'LSmallToe', 21: 'LHeel', 
                         22: 'RBigToe', 23: 'RSmallToe', 24: 'RHeel', 25: 'Background'}
    # Get also with print(op.getPosePartPairs(poseModel))
    part_pairs = [1, 8, 1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 10, 11, 8, 12, 12, 13, 13, 14, 1, 0, 0, 15,
                  15, 17, 0, 16, 16, 18, 2, 17, 5, 18, 14, 19, 19, 20, 14, 21, 11, 22, 22, 23, 11, 24]
    #Create a dict containing all end points (e.g. RWrist, LWrist)
    end_joints = dict()
    frequencies = collections.Counter(part_pairs)
    for (joint,freq) in frequencies.values():
        if freq == 1:
            end_joints.update({body_part_mapping[joint]})
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
    end_joints_depth2 = {4: 2, 7: 5, 17: 'REar', 18: 'LEar', 20: 'LSmallToe', 21: 'LHeel', 23: 'RSmallToe', 24: 'RHeel'}

    inv_body_part_mapping = dict((v,k) for k, v in body_part_mapping.iteritems())

    #Delete background label if present
    if 'Background' in inv_body_part_mapping:
        del xs[inv_body_part_mapping['Background']]
        del ys[inv_body_part_mapping['Background']]

    # Make all keypoints relative to a selected reference point
    ref_point = (xs[inv_body_part_mapping['MidHip']], ys[inv_body_part_mapping['MidHip']])
    keypoints = np.asarray(zip(xs,ys), dtype=np.float32)
    keypoints = keypoints - ref_point

    pose_descriptor = []
    #Joint Coordinate: all keypoints in relative coordinate system + reference point in world coordinate system
    #Dimension 25 x 2 + 1 = 51 
    joint_coordinates = np.delete(keypoints,inv_body_part_mapping['MidHip'])
    joint_coordinates = np.append(joint_coordinates, ref_point) 
    pose_descriptor.append(joint_coordinates.tolist())

    #Joint-Joint Distance
    #Dimension 25 over 2 = 300
    #scipy.special.binom(len(keypoints), 2))
    joint_distances = []
    #Only consider unique different joint pairs ,e.g. (1,2) == (2,1)
    for i1 in range(len(keypoints)):
        for i2 in range(len(keypoints)):
            if i1 < i2:
                joint_distances.append(numpy.linalg.norm(keypoints[i2]-keypoints[i1]))
    pose_descriptor.append(joint_distances)

    #Joint-Joint Orientation (vector orientation of unit vector from j1 to j2)
    #Dimension 25 over 2 = 300
    joint_orientations = []
    for i1 in range(len(keypoints)):
        for i2 in range(len(keypoints)):
            if i1 < i2:
                j1 = keypoints[i1]
                j2 = keypoints[i2]
                joint_orientations.append((j2-j1)/np.linalg.norm(j2-j1)) #a-b = target-origin
    pose_descriptor.append(joint_orientations)

    lines = []
    lines_adjacent_only = []
    #Directly adjacent lines, Dimesions: 25 - 1
    #Lines pointing outwards
    for i in range(len(keypoints)-1):
        lines.append(LineString([keypoints[i], keypoints[i+1]]))
        lines_adjacent_only.append(LineString([keypoints[i], keypoints[i+1]]))
    #Kinetic chain from end joints of depth 2
    for (end, begin_2) in end_joints_depth2.values():
        lines.append(LineString([keypoints[begin], keypoints[end]]))
    #Lines only between end joints
    for (k1,label1) in end_joints.values():
        for (k2,label2) in end_joints.values():
            if k1!=k2:
                lines.append(LineString([keypoints[k1], keypoints[k2]]))

    #Joint-Line Distance
    joint_line_distances = []
    for l in lines:
        coords = list(l.coords)
        for joint in keypoints:
            if joint not in coords:
                joint_line_distances.append(Point(joint).distance(l))
    pose_descriptor.append(joint_line_distances) 

    #Line-Line Angle
    #(25-1) over 2 = 276
    line_line_angles = []
    for l1 in lines_adjacent_only:
        for l2 in lines_adjacent_only:
            if l1!=l2:
                [p11,p12] = list(l1.coords)
                [p21,p22] = list(l2.coords)
                #limb vectors pointing outwards
                j1 = p12-p11
                j2 = p22-p21
                angle = np.arccos(np.dot(j1, j2) / (np.linalg.norm(j1) * np.linalg.norm(j2)))
                line_line_angles.append(angle)
    pose_descriptor.append(line_line_angles)

    #Planes for selected regions: 2 for each head, arms, leg & foot region 
    plane_points = [{1: 'Neck',0: 'Nose', 18: 'LEar'}, {1: 'Neck',0: 'Nose', 18: 'REar'}, {2: 'RShoulder', 3: 'RElbow', 4: 'RWrist'},
                    {5: 'LShoulder', 6: 'LElbow',7: 'LWrist'}, {12: 'LHip', 13: 'LKnee', 14: 'LAnkle'}, { 9: 'RHip', 10: 'RKnee', 11: 'RAnkle'},
                    {13: 'LKnee', 14: 'LAnkle', 19: 'LBigToe'}, {10: 'RKnee', 11: 'RAnkle', 22: 'RBigToe'}]
    planes = []
    for p in plane_points:
        indices = p.keys()
        planes.append(Polygon([keypoints(indices[0]),keypoints(indices[1]),keypoints(indices[2])]))
    
    #Joint-Plane Distance
    #Dimensions (25-3)*8 = 176
    joint_plane_distances = []
    for joint in keypoints:
        for plane in planes:
            if joint not in list(plane.exterior.coords):
                joint_plane_distances.append(Point(joint).distance(plane))
    pose_descriptor.append(joint_plane_distances)

    print(pose_descriptor)





    





