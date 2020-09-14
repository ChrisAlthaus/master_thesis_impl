import json
import h5py
import random
import numpy as np
from matplotlib.pyplot import imshow
from PIL import Image, ImageDraw
import datetime
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-predictdir')

args = parser.parse_args()

# load the following to files from DETECTED_SGG_DIR
pred_dir = os.path.join(args.predictdir, 'custom_prediction.json')
info_dir = os.path.join(args.predictdir, 'custom_data_info.json')
custom_prediction = json.load(open(pred_dir))
custom_data_info = json.load(open(info_dir))

def draw_single_box(pic, box, color='red', draw_info=None):
    draw = ImageDraw.Draw(pic)
    x1,y1,x2,y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    draw.rectangle(((x1, y1), (x2, y2)), outline=color)
    if draw_info:
        draw.rectangle(((x1, y1), (x1+50, y1+10)), fill=color)
        info = draw_info
        draw.text((x1, y1), info)

def drawline(pic, box1, box2, color='blue'):
    draw = ImageDraw.Draw(pic)
    x11,y11,x12,y12 = int(box1[0]), int(box1[1]), int(box1[2]), int(box1[3])
    m1x, m1y = (x11+x12)/2, (y11+y12)/2

    x21,y21,x22,y22 = int(box2[0]), int(box2[1]), int(box2[2]), int(box2[3])
    m2x, m2y = (x21+x22)/2, (y21+y22)/2
    #draw.line([(m1x,m1y), (m2x,m2y)], fill=color)
    draw.line([(x11,y11), (x21,y21)], fill=color)



        
def print_list(name, input_list, scores=None):
    for i, item in enumerate(input_list):
        if scores == None:
            print(name + ' ' + str(i) + ': ' + str(item))
        else:
            print(name + ' ' + str(i) + ': ' + str(item) + '; score: ' + str(scores[i]))
    
def draw_image(img_path, boxes, box_labels, rel_labels, rel_pairs, box_topk, rel_topk, box_scores=None, rel_scores=None):
    size = get_size(Image.open(img_path).size)
    pic = Image.open(img_path).resize(size)
    c = 0
    num_obj = len(boxes)
    for i in range(num_obj):
        if boxes[i] == 'INVALID':
            continue
        if c == box_topk:
            break
        info = str(i) + '_' + box_labels[i]
        draw_single_box(pic, boxes[i], draw_info=info)
        c= c+1

    c = 0
    num_rel = len(rel_pairs)
    print(boxes, len(boxes))
    for i in range(num_rel):
        id1, id2 = rel_pairs[i]

        print(id1,id2)
        b1 = boxes[id1]
        b2 = boxes[id2]
        if b1 == 'INVALID' or b2 == 'INVALID':
            continue
        drawline(pic, b1, b2)
        if c == rel_topk:
            break
        c = c + 1

    #display(pic)
    print("img path: ",img_path)
    print('*' * 50)
    print_list('box_labels', box_labels, box_scores)
    print('*' * 50)
    print_list('rel_labels', rel_labels, rel_scores)
    
    return pic

def get_size(image_size):
    min_size = 600
    max_size = 1000
    w, h = image_size
    size = min_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))
    if (w <= h and w == size) or (h <= w and h == size):
        return (w, h)
    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)
    return (ow, oh)

def filter(boxes, box_labels, box_scores, rel_pairs, rel_labels, rel_scores):
    ind_to_classes = ["__background__", "airplane", "animal", "arm", "bag", "banana", "basket",
                        "beach", "bear", "bed", "bench", "bike", "bird", "board", "boat", "book",
                        "boot", "bottle", "bowl", "box", "boy", "branch", "building", "bus", "cabinet",
                        "cap", "car", "cat", "chair", "child", "clock", "coat", "counter", "cow",
                        "cup", "curtain", "desk", "dog", "door", "drawer", "ear", "elephant",
                        "engine", "eye", "face", "fence", "finger", "flag", "flower", "food",
                        "fork", "fruit", "giraffe", "girl", "glass", "glove", "guy", "hair",
                        "hand", "handle", "hat", "head", "helmet", "hill", "horse", "house",
                        "jacket", "jean", "kid", "kite", "lady", "lamp", "laptop", "leaf",
                        "leg", "letter", "light", "logo", "man", "men", "motorcycle", "mountain",
                        "mouth", "neck", "nose", "number", "orange", "pant", "paper", "paw", "people",
                        "person", "phone", "pillow", "pizza", "plane", "plant", "plate", "player", "pole", "post",
                        "pot", "racket", "railing", "rock", "roof", "room", "screen", "seat", "sheep", "shelf",
                        "shirt", "shoe", "short", "sidewalk", "sign", "sink", "skateboard", "ski", "skier", "sneaker",
                        "snow", "sock", "stand", "street", "surfboard", "table", "tail", "tie", "tile", "tire", "toilet", "towel",
                        "tower", "track", "train", "tree", "truck", "trunk", "umbrella", "vase", "vegetable", "vehicle", "wave", "wheel",
                        "window", "windshield", "wing", "wire", "woman", "zebra"]
    ind_to_predicates = ["__background__", "above", "across", "against", "along", "and", "at", "attached to",
                        "behind", "belonging to", "between", "carrying", "covered in", "covering", "eating", "flying in",
                        "for", "from", "growing on", "hanging from", "has", "holding", "in", "in front of", "laying on",
                        "looking at", "lying on", "made of", "mounted on", "near", "of", "on", "on back of", "over",
                        "painted on", "parked on", "part of", "playing", "riding", "says", "sitting on",
                        "standing on", "to", "under", "using", "walking in", "walking on", "watching", "wearing", "wears", "with"]

    filter_classes = ["__background__", "animal", "bag", "basket",
                        "beach", "bed", "bench", "bird", "boat", "boy", "building","cabinet",
                         "cat", "chair", "child", "cow",
                        "cup", "curtain", "desk", "dog", "door", "elephant",
                        "flower", "food",
                        "fruit", "giraffe", "girl", "guy",
                        "hill", "horse", "house",
                        "kid", "lady", "lamp","hat","cap",
                        "light", "man", "men", "mountain",
                        "people",
                        "person", "plant", "plate",
                        "rock", "roof", "room", "sheep", "shelf",
                        "sidewalk",
                        "snow", "stand", "street", "table",
                        "tower", "track", "train", "tree", "vase", "vegetable", "vehicle", "wave",
                        "window","woman", "zebra"]
    filter_inds = [ind_to_classes.index(elem) for elem in filter_classes]
    print("filter inds: ",filter_inds)
    del_ind = []
    for i, label in enumerate(box_labels):
        if label not in filter_inds:
            del_ind.append(i)
    for i in sorted(del_ind, reverse=True):
        box_labels[i] = 'INVALID'
        boxes[i] = 'INVALID'
        box_scores[i] = 'INVALID'

    del_ind = []  
    for k,[label1,label2] in enumerate(rel_pairs):
        if label1 not in filter_inds or label2 not in filter_inds:
            del_ind.append(k)
    for k in sorted(del_ind, reverse=True):       
        del rel_pairs[k]
        del rel_scores[k]
        del rel_labels[k]



output_dir = os.path.join('/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/out/visualize', datetime.datetime.now().strftime('%m/%d_%H-%M-%S'))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
else:
    raise ValueError("Output directory %s already exists."%output_dir)

# parameters
#image_idx = 11
box_topk = 10 # select top k bounding boxes
rel_topk = 20 # select top k relationships
ind_to_classes = custom_data_info['ind_to_classes']
ind_to_predicates = custom_data_info['ind_to_predicates']


for image_idx in range(len(custom_data_info['idx_to_files'])):
    image_path = custom_data_info['idx_to_files'][image_idx]
    bbox = custom_prediction[str(image_idx)]['bbox']
    bbox_labels = custom_prediction[str(image_idx)]['bbox_labels']
    bbox_scores = custom_prediction[str(image_idx)]['bbox_scores']
    all_rel_labels = custom_prediction[str(image_idx)]['rel_labels']
    all_rel_scores = custom_prediction[str(image_idx)]['rel_scores']
    all_rel_pairs = custom_prediction[str(image_idx)]['rel_pairs']
    #rel pairs values index box items by order
    print("box labels: ", bbox_labels)
    print("bbox: ", bbox)
    print("all_rel_pairs: ", all_rel_pairs)
    print("max: ", max([item[0]for item in all_rel_pairs]))
    print("max: ", max([item[1]for item in all_rel_pairs]))
    print("len: ",len(bbox))
    #exit(1)
    filter(bbox, bbox_labels, bbox_scores, all_rel_pairs, all_rel_labels, all_rel_scores)
    print("box labels: ", bbox_labels)
    #boxes = bbox[:box_topk]
    #box_labels = bbox_labels[:box_topk]
    #box_scores = bbox_scores[:box_topk]

    #print("box labels: ", box_labels)
    print("all rel labels: ", len(all_rel_labels))
    print(len(all_rel_labels), len(all_rel_scores), len(all_rel_pairs))

    for i in range(len(bbox_labels)):
        if bbox_labels[i] == 'INVALID':
            continue
        bbox_labels[i] = ind_to_classes[bbox_labels[i]]

    """rel_labels = []
    rel_scores = []
    for i in range(len(all_rel_pairs)):
        if all_rel_pairs[i][0] < box_topk and all_rel_pairs[i][1] < box_topk:
            rel_scores.append(all_rel_scores[i])
            label = str(all_rel_pairs[i][0]) + '_' + box_labels[all_rel_pairs[i][0]] + ' => ' + ind_to_predicates[all_rel_labels[i]] + ' => ' + str(all_rel_pairs[i][1]) + '_' + box_labels[all_rel_pairs[i][1]]
            rel_labels.append(label)
    """
    #rel_labels = rel_labels[:rel_topk]
    #rel_scores = rel_scores[:rel_topk]

    #print("boxes: ",boxes, len(boxes))
    #print("rels: ",all_rel_pairs)
    #rel_pairs_filtered = [rel for rel in all_rel_pairs if rel[0] < box_topk and rel[1] < box_topk][:rel_topk]
    #print("filtered: ",rel_pairs_filtered)
    img = draw_image(image_path, bbox, bbox_labels, all_rel_labels, all_rel_pairs, box_topk, rel_topk, box_scores=bbox_scores, rel_scores=all_rel_scores)
    imgname =  "%s_scenegraph.jpg"%os.path.splitext(os.path.basename(image_path))[0]
    img.save(os.path.join(output_dir, imgname))