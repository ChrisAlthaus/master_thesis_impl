import h5py

import argparse
import os
import json

import numpy as np
import random

from visualizeimgs import draw_image, draw_single_box, drawline
from validlabels import ind_to_classes, ind_to_predicates

parser = argparse.ArgumentParser()
parser.add_argument('-file', help='Path to the scene graph file.') #VG-SGG-with-attri.h5
parser.add_argument('-imageinfo', help='Path to the image info file.') #VG-SGG-with-attri.h5
parser.add_argument('-imagefolder', help='Path to the image folder.') #VG-SGG-with-attri.h5
parser.add_argument('-outputdir')

args, unknown = parser.parse_known_args()
#args = parser.parse_args()

f = h5py.File(args.file, 'r')

img_info = None
with open(args.imageinfo, "r") as f_meta:
    img_info = json.load(f_meta)


#img_info = [elem for elem in img_info if elem['image_id'] not in [1592, 1722, 4616, 4617]]
#Important: do not sort img_info, the given image_data.json is from approx. 4000 image not in right ordering


for i in range(0,len(f['img_to_first_box']),1000): #random.sample(range(len(f['img_to_first_box'])),20): #
    box_indstart = f['img_to_first_box'][i]
    box_indend = f['img_to_last_box'][i]

    labels = f['labels'][box_indstart:box_indend+1].flatten()


    boxes_512 = f['boxes_512'][box_indstart:box_indend+1]
    boxes_1024 = f['boxes_1024'][box_indstart:box_indend+1]
    attr = f['attributes'][box_indstart:box_indend+1]

    rel_indstart = f['img_to_first_rel'][i]
    rel_indend = f['img_to_last_rel'][i]

    rels = f['relationships'][rel_indstart:rel_indend+1]
    preds = f['predicates'][rel_indstart:rel_indend+1].flatten()

    image_path = os.path.join(args.imagefolder,"%s.jpg"%img_info[i]['image_id'])
    box_topk = 40
    rel_topk = 40

    print("number boxes: ",len(boxes_512))
    print("number rels: ",len(rels))
    #print("boxes: ",boxes_512)
    #print("labels: ",labels.flatten(), type(labels))
    #print("first 10: ",f['boxes_1024'][:10])
    #print("first 10: ",f['boxes_512'][:10])
    #print("boxes_1024: ",boxes_1024, type(boxes_1024))
    #print("rels labels: ",preds)

    boxes_512[:, :2] = boxes_512[:, :2] - boxes_512[:, 2:] / 2
    boxes_512[:, 2:] = boxes_512[:, :2] + boxes_512[:, 2:]
    boxes_1024[:, :2] = boxes_1024[:, :2] - boxes_1024[:, 2:] / 2
    boxes_1024[:, 2:] = boxes_1024[:, :2] + boxes_1024[:, 2:]

    BOX_SCALE = 512
    w, h = img_info[i]['width'], img_info[i]['height']
    #print("width= ",w, 'height= ',h)
    # important: recover original box from BOX_SCALE
    boxes_512 = boxes_512 / BOX_SCALE * max(w, h)
    img, ann_str = draw_image(image_path, boxes_512, labels, rels, preds,
                   box_topk, rel_topk,
                   ind_to_classes, ind_to_predicates,
                   box_indstart = box_indstart)
    #print('*' * 40 )
    #print("Image %d"%i)
    #print(ann_str)
    imgname =  "%s_scenegraph.jpg"%os.path.splitext(os.path.basename(image_path))[0]
    img.save(os.path.join(args.outputdir, imgname))
    #if i==20:
    #    exit(1)
