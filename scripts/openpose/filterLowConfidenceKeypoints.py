#Filters all detected persons which have a low average confidence for the keypoints.
#Filtered JSON keypoint files are written to an output directory.
import os
import json 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-inputDirectory',required=True,
                    help='Path to the input directory.')
parser.add_argument('-outputDirectory',required=True,
                    help='Path to the output that contains the resumes.')
parser.add_argument('--thresh', default= 0.1,
                    help='Threshold for the poses necessary average keypoints confidence.')
args = parser.parse_args()

if not os.path.isdir(args.inputDirectory):
    raise ValueError("No valid input directory.")
if not os.path.isdir(args.outputDirectory):
    raise ValueError("No valid output directory.")
if args.thresh < 0 or args.thresh >1:
    raise ValueError("Threshold must be between 0 and 1")

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
        confs = person["pose_keypoints_2d"][2::3] #each 3th value in keypoints array is the confidence
        average_conf = sum(confs)/len(confs)
        print(json_file_path, average_conf)
        if average_conf < args.thresh:
            delete.append(person)
    
    for del_person in delete:
        json_data["people"].remove(del_person)
    
    with open(os.path.join(args.outputDirectory,json_file), 'w') as f:
        json.dump(json_data, f)


        

    

