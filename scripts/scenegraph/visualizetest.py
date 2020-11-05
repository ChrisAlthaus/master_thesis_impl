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

import sys
sys.path.append('/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/maskrcnn_benchmark/data/datasets')
from visual_genome import VGDataset


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
        info = str(i) + '_' + ind_to_predicates[rel_labels[i]]
        drawline(pic, b1, b2, draw_info=info)
        #b1_label = ind_to_classes[box_labels[id1]]
       # b2_label = ind_to_classes[box_labels[id2]]
    return pic

    

parser = argparse.ArgumentParser()
parser.add_argument('-file', help='Path to the scene graph file.') #VG-SGG-with-attri.h5
parser.add_argument('-imageinfo', help='Path to the image info file.') #image_data.json
parser.add_argument('-labelmappings') 
parser.add_argument('-imagefolder', help='Path to the image folder.') #VG_100K
parser.add_argument('-outputdir')
parser.add_argument('-randomsample', action="store_true")
parser.add_argument('-firstn', action="store_true")

args, unknown = parser.parse_known_args()
#args = parser.parse_args()

output_dir = os.path.join(args.outputdir, datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# ---------------------------------- VG CLASS TESTING ---------------------------------------
vgdataset = VGDataset('train', args.imagefolder, args.file, args.labelmappings, args.imageinfo, filter_non_overlap=False, filter_empty_rels=False)
print(vgdataset.img_info[0])
print(vgdataset.filenames[0])
print(vgdataset.gt_boxes[0])
print(vgdataset.gt_classes[0])
print(vgdataset.gt_attributes[0])
print(vgdataset.relationships[0])

c_noann = 0
for (image, target, _) in vgdataset:
    if len(target)<1:
        print(_)
        print("no bbox")
        c_noann += 1
print("Number of no annotations: ",c_noann)
exit(1)
indices = list(range(0,len(vgdataset.filenames),1000))
imgids = []
for i in indices:
    #print(vgdataset.filenames[i])
    #continue
    rels = vgdataset.relationships[i][:,:2]
    #print(vgdataset.relationships[i][0], type(vgdataset.relationships[i][0]), vgdataset.relationships[i][0][2])
    preds = vgdataset.relationships[i][:,2]
    #print("--")
    #print("Index %d :"%i, vgdataset.filenames[i], vgdataset.gt_boxes[i], vgdataset.gt_classes[i], rels, preds)
    #print("--")
    w, h = vgdataset.img_info[i]['width'], vgdataset.img_info[i]['height']
    gtboxes = vgdataset.gt_boxes[i] / 1024 * max(w, h)
    img = draw_image(vgdataset.filenames[i], gtboxes, vgdataset.gt_classes[i], rels, preds,
               vgdataset.ind_to_classes, vgdataset.ind_to_predicates)
    imgname =  "%s_scenegraph1.jpg"%os.path.splitext(os.path.basename(vgdataset.filenames[i]))[0]
    img.save(os.path.join(output_dir, imgname))
    imgids.append(int(os.path.splitext(os.path.basename(vgdataset.filenames[i]))[0]))
    #print("")

print("Wrote visualizations to: ", output_dir)

# ---------------------------------- OWN DATALOADER COMPARISON ---------------------------------------
f = h5py.File(args.file, 'r')
img_info = None
with open(args.imageinfo, "r") as f_meta:
    img_info = json.load(f_meta)

#img_info = [elem for elem in img_info if elem['image_id'] not in [1592, 1722, 4616, 4617]]
#Important: do not sort img_info, the given image_data.json is from approx. 4000 image not in right ordering
exit(1)
for i in imgids: #indices:
    #print(img_info[i]['image_id'])
    #continue
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
    

    boxes_512[:, :2] = boxes_512[:, :2] - boxes_512[:, 2:] / 2
    boxes_512[:, 2:] = boxes_512[:, :2] + boxes_512[:, 2:]
    #print("boxes_1024: ",boxes_1024)
    boxes_1024[:, :2] = boxes_1024[:, :2] - boxes_1024[:, 2:] / 2
    boxes_1024[:, 2:] = boxes_1024[:, :2] + boxes_1024[:, 2:]
    #print("boxes_1024: ",boxes_1024)
    #print("--")
    #print("Index %d :"%i, img_info[i], boxes_1024, labels, rels, preds)
    #print("--")
    BOX_SCALE = 512
    w, h = img_info[i]['width'], img_info[i]['height']
    #print("width= ",w, 'height= ',h)
    # important: recover original box from BOX_SCALE
    boxes_512 = boxes_512 / BOX_SCALE * max(w, h)
    boxes_1024 = boxes_1024 / 1024 * max(w, h)
    img = draw_image(image_path, boxes_1024, labels, rels, preds,
                   ind_to_classes, ind_to_predicates,
                   box_indstart = box_indstart)
    #print('*' * 40 )
    #print("Image %d"%i)
    #print(ann_str)
    imgname =  "%s_scenegraph2.jpg"%os.path.splitext(os.path.basename(image_path))[0]
    img.save(os.path.join(output_dir, imgname))
   
print("Wrote visualizations to: ", output_dir)
