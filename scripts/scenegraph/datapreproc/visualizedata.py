import h5py

import argparse
import os
import json

import numpy as np
import random

from visualizeimgs import draw_single_box, drawline, get_size
from validlabels import ind_to_classes, ind_to_predicates
import datetime
from PIL import Image, ImageDraw

#Visualizes a part of the VG dataset for validation purposes
#e.g. run with:
#python3.6 visualizedata.py -file /home/althausc/nfs/data/vg/VG-SGG-with-attri.h5 -imageinfo /home/althausc/nfs/data/vg/image_data.json 
#                   -imagefolder /home/althausc/nfs/data/vg/VG_100K/VG_100K -outputdir /home/althausc/nfs/data/vg/visualize_test -firstn

def draw_image(img_path, boxes, box_labels, rel_pairs, rel_labels,
               ind_to_classes, ind_to_predicates, box_indstart = None):

    #size = get_size(Image.open(img_path).size)
    pic = Image.open(img_path)#.resize(size)

    for i in range(len(boxes)):
        info = '%d_%s'%(i, ind_to_classes[box_labels[i]])
        draw_single_box(pic, boxes[i], draw_info=info, validsize=pic.size)

    #relationship values not starting from 0, e.g. for visualizing scene graph data
    #for predictions rel values starting from 0 relative to box array length
    if box_indstart is not None:
        rel_pairs = np.array(rel_pairs) - box_indstart

    for i in range(len(rel_pairs)):
        id1, id2 = rel_pairs[i]
        b1 = boxes[id1]
        b2 = boxes[id2]
        drawline(pic, b1, b2)
        #b1_label = ind_to_classes[box_labels[id1]]
       # b2_label = ind_to_classes[box_labels[id2]]
    return pic
    

parser = argparse.ArgumentParser()
parser.add_argument('-file', help='Path to the scene graph file.') #VG-SGG-with-attri.h5
parser.add_argument('-imageinfo', help='Path to the image info file.') #image_data.json
parser.add_argument('-imagefolder', help='Path to the image folder.') #VG_100K
parser.add_argument('-outputdir')
parser.add_argument('-randomsample', action="store_true")
parser.add_argument('-firstn', action="store_true")

args, unknown = parser.parse_known_args()
#args = parser.parse_args()

f = h5py.File(args.file, 'r')

img_info = None
with open(args.imageinfo, "r") as f_meta:
    img_info = json.load(f_meta)


#img_info = [elem for elem in img_info if elem['image_id'] not in [1592, 1722, 4616, 4617]]
#Important: do not sort img_info, the given image_data.json is from approx. 4000 image not in right ordering

output_dir = os.path.join(args.outputdir, datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

indices = []  
if args.randomsample:
    indices = random.sample(list(range(0,len(f['img_to_first_box']))), 100)
elif args.firstn:
    indices = list(range(0,len(f['img_to_first_box'])))[:100]
else:
    indices = list(range(0,len(f['img_to_first_box']),1000))


for i in indices:
    print("i= ",i)
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
    img = draw_image(image_path, boxes_512, labels, rels, preds,
                   ind_to_classes, ind_to_predicates,
                   box_indstart = box_indstart)
    #print('*' * 40 )
    #print("Image %d"%i)
    #print(ann_str)
    imgname =  "%s_scenegraph.jpg"%os.path.splitext(os.path.basename(image_path))[0]
    img.save(os.path.join(output_dir, imgname))
   
print("Wrote visualizations to: ", output_dir)