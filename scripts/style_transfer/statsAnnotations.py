
import os
import json 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-jsonAnnotation',required=True,
                    help='Path to the input directory.')

args = parser.parse_args()

if not os.path.isfile(args.jsonAnnotation):
    raise ValueError("Input JSON Annotation file not exists.")


jsonAnnotationFile = args.jsonAnnotation
json_data = None

with open (jsonAnnotationFile, "r") as f:
    json_data = json.load(f)

images = json_data['images']
annotations = json_data['annotations']

print("Number of images in JSON: ",len(images))
print("Number of annotations in JSON: ",len(annotations))

imgs = dict()
for img in json_data['images']:
    imgs[img['id']] = img
print("Number of reduces image id's: ",len(imgs))


