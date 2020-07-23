
import os
import json 
import argparse
import copy
import time
import hashlib

parser = argparse.ArgumentParser()
parser.add_argument('-jsonAnnotation',required=True,
                    help='Path to the input directory.')
parser.add_argument('-styleTransferImDir',required=True,
                    help='Path to the directory of the style-transfered images.')
parser.add_argument('-outputDirectory','-o',required=True,
                    help='Path to the output that contains the resumes.')
#parser.add_argument('-oName','-oN',required=True,
#                    help='Name of the output JSON file name.')
args = parser.parse_args()

if not os.path.isdir(args.outputDirectory):
    raise ValueError("No valid output directory.")
if not os.path.isfile(args.jsonAnnotation):
    raise ValueError("Input JSON Annotation file not exists.")


jsonAnnotationFile = args.jsonAnnotation
json_data = None

with open (jsonAnnotationFile, "r") as f:
    json_data = json.load(f)

images = json_data['images']
image_annotations_updated = {'images':[]}
image_ids_added = []

#Style transfered images maybe not in person_keypoints file under subitem annotations, beacause sometimes no person present
#Also all images are present under subitem images
jsonFiles = [f for f in  os.listdir(args.styleTransferImDir) if os.path.isfile(os.path.join(args.styleTransferImDir, f))]

#Mapping for coco filename -> [styleimageName1,..,styleimageNameN]
cocoToStyle = dict()
for f in jsonFiles:
    #Content-Style image name syntax: filenameCOCO_filenameStyleImage.jpg
    if len(f.split('_')) == 1:
        raise ValueError("Image %s has not been style transfered."%f)

    coco_filename = f.split('_')[0]
    style_filename = os.path.splitext(os.path.basename(f.split('_')[1]))[0] 
    if coco_filename in cocoToStyle:
         cocoToStyle[coco_filename].append(style_filename)    
    else:
        cocoToStyle.update({coco_filename: [style_filename]})

#For every coco image get corresponding style-transfered images
for image_entry in images:
    filename = os.path.splitext(os.path.basename(image_entry['file_name']))[0]

    #If content coco image was used for style transfer, append all transfered image annotation entries
    if filename in cocoToStyle:
        for style_filename in cocoToStyle[filename]:
            styleImName = filename+'_'+style_filename
            print(styleImName)
            image_entry['file_name'] = str(int(hashlib.md5(styleImName.encode('utf-8')).hexdigest(), 16)) + '.jpg'
            image_entry['id'] = int(hashlib.md5(styleImName.encode('utf-8')).hexdigest(), 16)
            
            image_annotations_updated['images'].append(copy.copy(image_entry))
        image_ids_added.append(image_entry['id'])
        
    

#Reduce full annotation list to images added previously
annotations = json_data['annotations']
annotations_updated = {'annotations':[]}
#Map: original image_id -> {indices of original annotations}
imId_to_index = dict()
for i in range(len(annotations)):
    imId = annotations[i]['image_id']
    if imId in imId_to_index:
        imId_to_index[imId].append(i)
    else:
        imId_to_index.update({imId:[i]})

#Add for each content-style image an annotation entry with add. saved filename (c_s.jpg)
#in following order for cocoapi preprocessing
#Ordering for annotations is given by cocoapi: [c1_s1, c1_s1, c1_s1, c1_s2, c1_s2, c1_s2, c2_s3, c2_s4, ...]
for i in range(len(annotations)):
    image_id = annotations[i]['image_id']
    content_filename = "%012d"%int(image_id)
    if content_filename in cocoToStyle:
        for style_filename in cocoToStyle[content_filename]:
            annotations_of_image_id = []
            for i in imId_to_index[int(image_id)]:
                annotation_entry = annotations[i]
                styleImName = content_filename+'_'+style_filename
                print(styleImName)
                annotation_entry.update({'image_id':int(hashlib.md5(styleImName.encode('utf-8')).hexdigest(), 16)})
                annotations_of_image_id.append(copy.copy(annotation_entry))
            annotations_updated['annotations'].append(copy.copy(annotations_of_image_id))
        

"""for i in range(len(annotations)):
    annotation_entry = annotations[i]
    image_id = annotation_entry['image_id']
    if image_id in image_ids_added:
        annotations_image_id = [annotation_entry]   #e.g. because one annotation entry for every person on an image
        iAdd = 0
        for j in range(i+1,len(annotations)):
            #print(type(annotations[j]['image_id']),type(image_id))
            if image_id == 139:
                print(i,j,annotations[j])
            if annotations[j]['image_id'] == image_id:
                print("found multiple annotations for image id: ",image_id)
                annotations_image_id.append(annotations[j])
                iAdd = iAdd + 1
            else:
                break
              
        content_filename = "%012d"%int(annotation_entry['image_id']) # TODO: string without .jpg
        for style_filename in cocoToStyle[content_filename]:
            for annotation_entry in annotations_image_id:
                annotation_entry.update({'style_id': int(style_filename)})
                annotations_updated['annotations'].append(copy.copy(annotation_entry))
        #if image_id == 139:
        #    print(annotations_image_id)
        #    exit(1)
        i = i + iAdd"""
    
#Replace with modified image descriptor annotations & annotations 
del json_data['images']
del json_data['annotations']
json_data.update(image_annotations_updated)
json_data.update(annotations_updated)


outfile_name = os.path.splitext(os.path.basename(args.jsonAnnotation))[0] +'_st.json'
with open(os.path.join(args.outputDirectory,outfile_name), 'w') as f:
    json.dump(json_data, f, separators=(', ', ': '))

