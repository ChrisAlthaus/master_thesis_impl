
import os
import json 
import argparse
import copy
import time
import hashlib
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('-jsonAnnotation',required=True,
                    help='Path to the input directory.')
parser.add_argument('-styleTransferImDir',required=True,
                    help='Path to the directory of the style-transfered images.')
parser.add_argument('-outputDirectory','-o',required=True,
                    help='Path to the output that contains the resumes.')
parser.add_argument('-annotationSchema','-annSchema',required=True,type=str,
                    help='Schema 1: For each style-transfered image in the image field an unique annotation entry will be generated (used for OpenPose). \
                          Schema 2: For each content image of the style-tranfered image an unique annotation entry will be generated. (normal mode) \
                          Set \'COCOAPI\' for Schema 1 or \'Normal\' for Schema 2.')
#parser.add_argument('-oName','-oN',required=True,
#                    help='Name of the output JSON file name.')
args = parser.parse_args()

if not os.path.isdir(args.outputDirectory):
    raise ValueError("No valid output directory.")
if not os.path.isfile(args.jsonAnnotation):
    raise ValueError("Input JSON Annotation file not exists.")

annTypes = ['COCOAPI','Normal']
if args.annotationSchema not in annTypes:
    raise ValueError("No valid annotation schema entered.")


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
print("Number of input images: ", len(jsonFiles))

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
            im = Image.open(os.path.join(args.styleTransferImDir, styleImName + '.jpg'))
            width, height = im.size
            print("Adding image descriptor for: ",styleImName)
            image_entry['file_name'] = styleImName + '.jpg'
            
            
            image_entry['width'] = width
            image_entry['height'] = height
            #image_entry['id'] = int(hashlib.md5(styleImName.encode('utf-8')).hexdigest(), 16)
            if(args.annotationSchema == 'COCOAPI'):
                image_entry['id'] = int("%s%s"%(filename,style_filename))
            
            image_annotations_updated['images'].append(copy.copy(image_entry))
        image_ids_added.append(image_entry['id'])
        
#Reduce full annotation list to images added previously
annotations = json_data['annotations']
annotations_updated = {'annotations':[]}    



#Output Json can be processed by COCO API, where ids of images have to be unique
if(args.annotationSchema == 'COCOAPI'):
    #Map: original image_id -> {indices of original annotations}
    imId_to_index = dict()
    for i in range(len(annotations)):
        imId = annotations[i]['image_id']
        if imId in imId_to_index:
            imId_to_index[imId].append(i)
        else:
            imId_to_index.update({imId:[i]})

    #Add for each content-style image an annotation entry with add. saved filename (c_s.jpg) ?
    #in following order for cocoapi preprocessing (ordering necessary for OpenPose) ?
    #Ordering for annotations is given by cocoapi: [c1_s1, c1_s1, c1_s1, c1_s2, c1_s2, c1_s2, c2_s3, c2_s4, ...]
    ann_id_num = 1  #also unqiue annotation id (!= annotation image_id!)
    c = 0
    #Approach: Search for every coco annotation the corresponding content-style image names
    #Expand the annotation to multiple annotations matching all content-style ids
    for i in range(len(annotations)):
        image_id = annotations[i]['image_id']
        content_filename = "%012d"%int(image_id)
        #If coco imgid in image input directory
        if content_filename in cocoToStyle:
            for style_filename in cocoToStyle[content_filename]:
                annotations_of_image_id = []
                for i in imId_to_index[int(image_id)]:
                    annotation_entry = annotations[i]
                    styleImName = content_filename+'_'+style_filename
                    #print("Adding annotation for: ",styleImName)
                    #print(int("%s%s"%(content_filename,style_filename)))
                    #annotation_entry.update({'image_id':int(hashlib.md5(styleImName.encode('utf-8')).hexdigest(), 16)})
                    annotation_entry.update({'image_id':int("%s%s"%(content_filename, style_filename))})
                    annotation_entry.update({'id':ann_id_num})
                    
                    annotations_of_image_id.append(copy.copy(annotation_entry))
                    ann_id_num = ann_id_num + 1
                annotations_updated['annotations'].extend(copy.copy(annotations_of_image_id))
                c += 1
        if c%1000 == 0:
            print("Processed %d images"%c)

#Ids of images are not unique, so that only for the base/content image the annoations are saved
#Half the memory usage of actual unique annotations
elif(args.annotationSchema == 'Normal'):
        for i in range(len(annotations)):
            image_id = annotations[i]['image_id']
            content_filename = "%012d"%int(image_id)
            if content_filename in cocoToStyle:           
                annotations_updated['annotations'].append(copy.copy( annotations[i]))
  
#Replace with modified image descriptor annotations & annotations 
del json_data['images']
json_data.update(image_annotations_updated)
del json_data['annotations']
json_data.update(annotations_updated)

if(args.annotationSchema == 'COCOAPI'):
    outfile_name = os.path.splitext(os.path.basename(args.jsonAnnotation))[0] +'_stAPI.json'
else:
    outfile_name = os.path.splitext(os.path.basename(args.jsonAnnotation))[0] +'_stNorm.json'

with open(os.path.join(args.outputDirectory,outfile_name), 'w') as f:
    json.dump(json_data, f, separators=(', ', ': '))

