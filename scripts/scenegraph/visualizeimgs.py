import json
import h5py
import random
import numpy as np
from matplotlib.pyplot import imshow
from PIL import Image, ImageDraw
import datetime
import os

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-predictdir')
    parser.add_argument('-filter', action='store_true', 
                        help='Specify if boxes should be filtered by predefined labels. Filtered out boxes will be shown in green.')

    args = parser.parse_args()

    # load the following to files from DETECTED_SGG_DIR
    pred_dir = os.path.join(args.predictdir, 'custom_prediction.json')
    info_dir = os.path.join(args.predictdir, 'custom_data_info.json')
    custom_prediction = json.load(open(pred_dir))
    custom_data_info = json.load(open(info_dir))

def get_filterinds():
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
    return filter_inds


def draw_single_box(pic, box, color='red', draw_info=None, validsize=None):
    draw = ImageDraw.Draw(pic)
    #print("draw: ",box)
    x1,y1,x2,y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    if validsize != None:
        if x1>validsize[0] or x2>validsize[0] or y1>validsize[1] or y2>validsize[1]:
            print("Box partially not on image: ", box)
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
    
def draw_image(img_path, boxes, box_labels, rel_pairs, rel_labels, box_topk, rel_topk,
               ind_to_classes, ind_to_predicates, box_scores=None, rel_scores=None, filter=False, box_indstart = None):

    size = get_size(Image.open(img_path).size)
    print("resize: ",size)
    pic = Image.open(img_path).resize(size)
    ann_str = ''

    if filter:  #apply class filter
        validlabels = get_filterinds()
    else:
        validlabels = [ind_to_classes.index(elem) for elem in ind_to_classes]

    addedlabels = []    #remember added box index to the image
                        #to filter valid relations (relations having (boxid1,boxid2) items)
    ann_str = ann_str + 'Box labels: \n'
    c = 0
    num_obj = len(boxes)
    #print("boxes: ",boxes)
    #print("rel_pairs: ",rel_pairs)
    #print("box_labels: ",box_labels)
    for i in range(num_obj):
        if c == box_topk:
            break
        if box_labels[i] in validlabels:
            info = str(i) + '_' + ind_to_classes[box_labels[i]]
            draw_single_box(pic, boxes[i], draw_info=info, validsize=size)
            addedlabels.append(i)
            
            if box_scores is not None:
                info = info + "; " + str(box_scores[i])
            ann_str = ann_str + info + '\n'
            c = c + 1
        else:
            #draw bounding box of class not considered which acutally be in top-k
            info = str(i) + '_' + ind_to_classes[box_labels[i]]
            draw_single_box(pic, boxes[i], draw_info=info, color='green')

    ann_str = ann_str + 'Rel labels: \n'
    c = 0        
    num_rel = len(rel_pairs)
    #print("rel_pairs: ",rel_pairs)
    #relationship values not starting from 0, e.g. for visualizing scene graph data
    #for predictions rel values starting from 0 relative to box array length
    if box_indstart is not None:
        rel_pairs = np.array(rel_pairs) - box_indstart
    #print(box_indstart, rel_pairs)
   #print(rel_pairs)
    #print("rel_pairs: ",rel_pairs)
    #print(len(rel_pairs))
    #print("rel_labels: ",rel_labels)
    #print("addedlabels: ",addedlabels)
    for i in range(len(rel_pairs)):
        #print(i ,c ,rel_topk)
        if c == rel_topk:
            break
        id1, id2 = rel_pairs[i]
        #print(rel_pairs[i])
        #if id1 in [13,1] and id2 in [13,1]:
        #    print("found: ",id1,id2)

        if filter:
            if id1 not in addedlabels or id2 not in addedlabels:
                continue
        if id1 in addedlabels and id2 in addedlabels:
            b1 = boxes[id1]
            b2 = boxes[id2]

            drawline(pic, b1, b2)
            b1_label = ind_to_classes[box_labels[id1]]
            b2_label = ind_to_classes[box_labels[id2]]
            relstr = str(id1) + '_' + b1_label + ' => ' + ind_to_predicates[rel_labels[i]] + ' => ' + str(id2) + '_' + b2_label
            if rel_scores is not None:
                relstr = relstr + '; ' + str(rel_scores[i]) + '\t%d'%i
            
            ann_str = ann_str + relstr + '\n'
            c = c + 1
    
 
    if filter: 
        ann_str = ann_str + 'Actual Top-k rel: \n'
        for i in range(rel_topk):
            id1, id2 = rel_pairs[i]

            b1 = boxes[id1]
            b2 = boxes[id2]
            b1_label = ind_to_classes[box_labels[id1]]
            b2_label = ind_to_classes[box_labels[id2]]
            relstr = str(id1) + '_' + b1_label + ' => ' + ind_to_predicates[rel_labels[i]] + ' => ' + str(id2) + '_' + b2_label
            if rel_scores is not None:
                relstr = relstr + '; ' + str(rel_scores[i]) + '\t%d'%i
            
            ann_str = ann_str + relstr + '\n'

    """print("img path: ",img_path)
    print('*' * 50)
    print_list('box_labels', box_labels, box_scores)
    print('*' * 50)
    print_list('rel_labels', rel_labels, rel_scores)"""
    
    return pic, ann_str

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


if __name__ == "__main__":
    output_dir = os.path.join('/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/out/visualize', datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        raise ValueError("Output directory %s already exists."%output_dir)

    # parameters
    #image_idx = 11
    box_topk = 20 # select top k bounding boxes
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
    
        img, ann_str = draw_image(image_path, bbox, bbox_labels, all_rel_pairs, all_rel_labels,
                        box_topk, rel_topk,
                        ind_to_classes, ind_to_predicates, box_scores=bbox_scores, rel_scores=all_rel_scores, filter=args.filter)
        imgname =  "1%s_scenegraph.jpg"%os.path.splitext(os.path.basename(image_path))[0]
        img.save(os.path.join(output_dir, imgname))

        annname = "2%s_labels.txt"%os.path.splitext(os.path.basename(image_path))[0]
        with open(os.path.join(output_dir, annname), "w") as text_file:
            text_file.write(ann_str)

    print("Saved visualizations to: ", output_dir)