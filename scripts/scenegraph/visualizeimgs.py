import json
import h5py
import random
import numpy as np
from matplotlib.pyplot import imshow
from PIL import Image, ImageDraw, ImageFont
import datetime
import os
import random

import argparse
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from validlabels import ind_to_classes, ind_to_predicates, VALID_BBOXLABELS

#Used for visualizing all graph predictions given by inputfiles (custom_prediction.json, custom_data_info.json).
#Save image overlayed by graphs (labeled bboxes and relationship). Additional save bbox and relationship annotations
#as easy-readable text to file.

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-predictdir')
    parser.add_argument('-filterlabels', action='store_true', 
                        help='Specify if boxes should be filtered by predefined labels. Filtered out boxes will be shown in green.\
                              Not necessary when filtered previously.')
    parser.add_argument('-boxestopk', default=-1, type=int, help='Only draw best scored first k boxes')
    parser.add_argument('-relstopk', default=-1, type=int, help='Only draw best scored first k relationships')
    parser.add_argument('-boxestresh', default=0.0, type=float)
    parser.add_argument('-relstresh', default=0.0, type=float)
    parser.add_argument('-visrandom', action='store_true', help='Only draw random subset of predictions.')
    args = parser.parse_args()

    # load the following to files from DETECTED_SGG_DIR
    pred_dir = os.path.join(args.predictdir, 'custom_prediction.json')
    info_dir = os.path.join(args.predictdir, 'custom_data_info.json')
    custom_prediction = json.load(open(pred_dir))
    custom_data_info = json.load(open(info_dir))


    #output_dir = os.path.join('/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/out/visualize', datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
    output_dir = os.path.join(args.predictdir, '.visimages')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        print("Warning: Output directory %s already exists."%output_dir)

    box_topk = args.boxestopk # select top k bounding boxes
    rel_topk = args.relstopk # select top k relationships
    box_tresh = args.boxestresh
    rel_tresh = args.relstresh
    ind_to_classes = custom_data_info['ind_to_classes']
    ind_to_predicates = custom_data_info['ind_to_predicates']

    c = 0
    if args.visrandom:
        ids = random.sample(list(range(len(custom_data_info['idx_to_files']))), 1000)
    else:
        ids = list(range(len(custom_data_info['idx_to_files'])))

    for image_idx in ids:
        image_path = custom_data_info['idx_to_files'][image_idx]
        if str(image_idx) not in custom_prediction:
            continue
        bbox = custom_prediction[str(image_idx)]['bbox']
        bbox_labels = custom_prediction[str(image_idx)]['bbox_labels']
        bbox_scores = custom_prediction[str(image_idx)]['bbox_scores']
        all_rel_labels = custom_prediction[str(image_idx)]['rel_labels']
        all_rel_scores = custom_prediction[str(image_idx)]['rel_scores']
        all_rel_pairs = custom_prediction[str(image_idx)]['rel_pairs']
    
        img, ann_str = draw_image(image_path, bbox, bbox_labels, all_rel_pairs, all_rel_labels,
                        box_topk, rel_topk, box_scores=bbox_scores, rel_scores=all_rel_scores, filterlabels=args.filterlabels, 
                        box_tresh=box_tresh, rel_tresh = rel_tresh)
        imgname =  "%s_1scenegraph.png"%os.path.splitext(os.path.basename(image_path))[0]
        #if img.mode != 'RGB':
        img = img.convert('RGBA')
        img.save(os.path.join(output_dir, imgname))

        annname = "%s_2labels.txt"%os.path.splitext(os.path.basename(image_path))[0]
        with open(os.path.join(output_dir, annname), "w") as text_file:
            text_file.write(ann_str)

        c = c + 1
        if c%100 == 0:
            print("Processed {} predictions.".format(c))

    print("Saved visualizations to: ", output_dir)


def get_filterinds():
    filter_inds = [ind_to_classes.index(elem) for elem in VALID_BBOXLABELS]
    return filter_inds


def draw_single_box(pic, box, color='red', draw_info=None, validsize=None):
    color = (255,0,0,200)
    draw = ImageDraw.Draw(pic, "RGBA")
    #print("draw: ",box)
    x1,y1,x2,y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    if validsize != None:
        if x1>validsize[0] or x2>validsize[0] or y1>validsize[1] or y2>validsize[1]:
            print("Box partially not on image: ", box)
    draw.rectangle(((x1, y1), (x2, y2)), outline=color)
    
    font = ImageFont.truetype("/usr/share/fonts/liberation-fonts/LiberationSans-Bold.ttf", 18, encoding="unic")
    if draw_info:
        draw.rectangle(((x1, y1), (x1+font.getsize(draw_info)[0], y1+font.getsize(draw_info)[1])), fill=color)
        info = draw_info
        
        draw.text((x1, y1), info, font=font)

def drawline(pic, box1, box2, color='blue', draw_info=None):
    color = (0,0,255,200)
    draw = ImageDraw.Draw(pic)
    x11,y11,x12,y12 = int(box1[0]), int(box1[1]), int(box1[2]), int(box1[3])
    m1x, m1y = (x11+x12)/2, (y11+y12)/2

    x21,y21,x22,y22 = int(box2[0]), int(box2[1]), int(box2[2]), int(box2[3])
    m2x, m2y = (x21+x22)/2, (y21+y22)/2
    #draw.line([(m1x,m1y), (m2x,m2y)], fill=color)
    draw.line([(x11,y11), (x21,y21)], fill=color)
    
    font = ImageFont.truetype("/usr/share/fonts/liberation-fonts/LiberationSans-Bold.ttf", 18, encoding="unic")
    if draw_info:
        linemx = (x11+x21)/2
        linemy = (y11+y21)/2
        draw.rectangle(((linemx, linemy), (linemx+font.getsize(draw_info)[0], linemy+font.getsize(draw_info)[1])), fill=color)
        
        draw.text((linemx, linemy), draw_info, font=font)


   
def print_list(name, input_list, scores=None):
    for i, item in enumerate(input_list):
        if scores == None:
            print(name + ' ' + str(i) + ': ' + str(item))
        else:
            print(name + ' ' + str(i) + ': ' + str(item) + '; score: ' + str(scores[i]))

    
def draw_image(imagesrc, boxes, box_labels, rel_pairs, rel_labels, box_topk=None, rel_topk=None, 
                box_scores=None, rel_scores=None, filterlabels=False, box_indstart = None, box_tresh= 0.0, rel_tresh = 0.0):
    #Draw the scene graph onto the input image. 
    #Additional options which can be used:
    #   - Filter out invalid labels from the input scene graph annotation
    #   - Only draw the top k bounding boxes
    #   - Only draw the top k relationships
    #Returns: overlayed image & 
    #         string with additional information regarding class and relationship labels

    if isinstance(imagesrc, str):
        pic = Image.open(imagesrc)
    elif isinstance(imagesrc, np.ndarray):
        pic = Image.fromarray(imagesrc.astype('uint8'), 'RGB')
    elif isinstance(imagesrc, Image.Image):
        pic = imagesrc
    else:
        raise ValueError("Input image source {} not supported.".format(type(imagesrc)))

    #if pic.mode in ("RGBA", "P"):
    #    pic = pic.convert("RGB")
    pic = pic.convert("RGBA")

    ann_str = ''
    if filterlabels:  #apply class filter
        validlabels = get_filterinds()
    else:
        validlabels = [ind_to_classes.index(elem) for elem in ind_to_classes]

    addedlabels = []    #remember added box index to the image
                        #to filter valid relations (relations having (boxid1,boxid2) items)
   
    #For prediction rescaling regarding configs min/max size is necessary
    size = get_size(pic.size)
    #print("Resize to: ",size)
    pic = pic.resize(size)
    #print(pic.size)

    #draw bboxes
    ann_str = ann_str + 'Box labels: \n'
    c = 0
    num_obj = len(boxes)
    if box_topk == -1:
        box_topk = len(boxes)

    for i in range(num_obj):
        if c == box_topk:
            break
        if box_labels[i] in validlabels and box_scores[i]>= box_tresh:
            info = str(i) + '_' + ind_to_classes[box_labels[i]]
            draw_single_box(pic, boxes[i], draw_info=info, color='red', validsize=pic.size)
            addedlabels.append(i)
            
            if box_scores is not None:
                info = info + "; " + str(box_scores[i])
            ann_str = ann_str + info + '\n'
            c = c + 1
        #else:
        #    #draw bounding box of class not considered which acutally be in top-k
        #    info = str(i) + '_' + ind_to_classes[box_labels[i]]
        #    draw_single_box(pic, boxes[i], draw_info=info, color='green')

    ann_str = ann_str + 'Rel labels: \n'
    c = 0        
    num_rel = len(rel_pairs)

    # draw relationships
    # relationship values not starting from 0, e.g. for visualizing scene graph data
    #for predictions rel values starting from 0 relative to box array length
    if box_indstart is not None:
        rel_pairs = np.array(rel_pairs) - box_indstart
    if rel_topk == -1:
        rel_topk = len(rel_pairs)

    for i in range(len(rel_pairs)):
        if c == rel_topk:
            break
        if rel_scores[i]<rel_tresh:
            continue
        id1, id2 = rel_pairs[i]
        b1 = boxes[id1]
        b2 = boxes[id2]
        b1_label = ind_to_classes[box_labels[id1]]
        b2_label = ind_to_classes[box_labels[id2]]
        
        if id1 not in addedlabels or id2 not in addedlabels:
            continue
        if filterlabels:
            if id1 in addedlabels and id2 in addedlabels:
                info = str(i) + '_' + ind_to_predicates[rel_labels[i]]
                print('draw lin rel info: ', info)
                drawline(pic, b1, b2, draw_info=info)
                relstr = str(id1) + '_' + b1_label + ' => ' + ind_to_predicates[rel_labels[i]] + ' => ' + str(id2) + '_' + b2_label
                if rel_scores is not None:
                    relstr = relstr + '; ' + str(rel_scores[i]) + '\t%d'%i
                
                ann_str = ann_str + relstr + '\n'
                c = c + 1
        else:
            info = str(i) + '_' + ind_to_predicates[rel_labels[i]]
            drawline(pic, b1, b2, draw_info=info)
            relstr = str(id1) + '_' + b1_label + ' => ' + ind_to_predicates[rel_labels[i]] + ' => ' + str(id2) + '_' + b2_label
            if rel_scores is not None:
                relstr = relstr + '; ' + str(rel_scores[i]) + '\t%d'%i
            
            ann_str = ann_str + relstr + '\n'
            c = c + 1

    #Print not filtered top-k relationships
    if filterlabels: 
        ann_str = ann_str + 'Actual Top-k rel: \n'
        #print(len(rel_pairs), rel_topk)
        for i in range(min(len(rel_pairs), rel_topk)):
            id1, id2 = rel_pairs[i]

            b1 = boxes[id1]
            b2 = boxes[id2]
            b1_label = ind_to_classes[box_labels[id1]]
            b2_label = ind_to_classes[box_labels[id2]]
            relstr = str(id1) + '_' + b1_label + ' => ' + ind_to_predicates[rel_labels[i]] + ' => ' + str(id2) + '_' + b2_label
            if rel_scores is not None:
                relstr = relstr + '; ' + str(rel_scores[i]) + '\t%d'%i
            
            ann_str = ann_str + relstr + '\n'
    
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
    main()