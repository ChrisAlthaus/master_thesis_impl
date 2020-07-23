#Prints statistics for coco annoations file for the purpose of validation
import os
import json 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-jsonAnnotation',required=True,
                    help='Path to the input directory.')
args = parser.parse_args()

with open (args.jsonAnnotation, "r") as f:
    json_data = json.load(f)

images = json_data['images']

print("Number of image entries: ",len(images))
for im in images:
    print(im['file_name'], im['id'])