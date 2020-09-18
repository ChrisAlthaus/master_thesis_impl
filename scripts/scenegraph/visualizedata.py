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
with open(args.imageinfo, "r") as f:
    img_info = json.load(f)


for i in random.sample(range(len(f['img_to_first_box'])),10):
    box_indstart = f['img_to_first_box'][i]
    box_indend = f['img_to_last_box'][i]

    labels = f['labels'][box_indstart:box_indend+1]


    boxes_512 = f['boxes_512'][box_indstart:box_indend+1]
    boxes_1024 = f['boxes_1024'][box_indstart:box_indend+1]
    attr = f['attributes'][box_indstart:box_indend+1]

    rel_indstart = f['img_to_first_rel'][i]
    rel_indend = f['img_to_last_rel'][i]

    rels = list(f['relationships'][rel_indstart:rel_indend+1])
    preds = list(f['predicates'][rel_indstart:rel_indend+1])

    image_path = os.path.join(args.imagefolder,"%s.jpg"%img_info[i])
    box_topk = 10
    rel_topk = 20

    img, ann_str = draw_image(image_path, boxes_512, labels, rels, preds,
                   box_topk, rel_topk,
                   ind_to_classes, ind_to_predicates)
    imgname =  "%s_scenegraph.jpg"%os.path.splitext(os.path.basename(image_path))[0]
    img.save(os.path.join(args.outputdir, imgname))
