#Creates custom features from skeleton pose annotations based on geometric properties.
import os
import json 
import argparse
import numpy as np

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

    delete = []
    for person in json_data["people"]:
        keypoints_x = person["pose_keypoints_2d"][::3]
        keypoints_y = person["pose_keypoints_2d"][1::3]

        keypoint_feature = calculateGPD(keypoints_x,keypoints_y)

    
    with open(os.path.join(args.outputDirectory,json_file), 'w') as f:
        json.dump(json_data, f)


def calculateGPD(xs,ys):
    body_part_mapping = {0: 'Nose', 1: 'Neck', 2: 'RShoulder', 3: 'RElbow', 4: 'RWrist', 5: 'LShoulder', 6: 'LElbow',
                         7: 'LWrist', 8: 'MidHip', 9: 'RHip', 10: 'RKnee', 11: 'RAnkle', 12: 'LHip', 13: 'LKnee', 14: 'LAnkle',
                         15: 'REye', 16: 'LEye', 17: 'REar', 18: 'LEar', 19: 'LBigToe', 20: 'LSmallToe', 21: 'LHeel', 
                         22: 'RBigToe', 23: 'RSmallToe', 24: 'RHeel', 25: 'Background'}
    inv_body_part_mapping = dict((v,k) for k, v in body_part_mapping.iteritems())

    # Make all keypoints relative to a selected reference point
    ref_point = (xs[inv_body_part_mapping['MidHip']], ys[inv_body_part_mapping['MidHip']])
    keypoints = np.asarray(zip(xs,ys), dtype=np.float32)
    keypoints = keypoints - ref_point

    pose_descriptor = []
    #Joint Coordinate: all keypoints in relative coordinate system + reference point in world coordinate system
    #Dimension #TODO: remove last item 25
    joint_coordinates = np.delete(keypoints,inv_body_part_mapping['MidHip'])
    joint_coordinates = np.append(joint_coordinates, ref_point) 
    #Joint-Joint Distance
