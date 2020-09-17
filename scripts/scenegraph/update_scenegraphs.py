import h5py

import argparse
import os
import json

import numpy as np

from validlabels import VALID_LABELS

parser = argparse.ArgumentParser()
parser.add_argument('-file', help='Path to the scene graph file.') #VG-SGG-with-attri.h5
parser.add_argument('-outputdir')

args = parser.parse_args()

f = h5py.File(args.file, 'r')

def get_validinds(labels, ind_to_classes, type):
    valid_inds = [ind_to_classes.index(elem) for elem in VALID_LABELS]
    if type == 'bbox':
        mask = [True if label in valid_inds else False for label in labels]
    elif type == 'rel':
        mask = [True if (rel[0] in valid_inds and rel[1] in valid_inds) else False for rel in labels]
    
    return mask

def load_info(dict_file, add_bg=True):
    """
    Loads the file containing the visual genome label meanings
    """
    info = json.load(open(dict_file, 'r'))
    if add_bg:
        info['label_to_idx']['__background__'] = 0
        info['predicate_to_idx']['__background__'] = 0
        info['attribute_to_idx']['__background__'] = 0

    class_to_ind = info['label_to_idx']
    predicate_to_ind = info['predicate_to_idx']
    attribute_to_ind = info['attribute_to_idx']
    ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
    ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])
    ind_to_attributes = sorted(attribute_to_ind, key=lambda k: attribute_to_ind[k])

    return ind_to_classes, ind_to_predicates, ind_to_attributes

ind_to_classes, ind_to_predicates, ind_to_attributes = load_info('/home/althausc/nfs/data/vg_styletransfer/VG-SGG-dicts-with-attri.json')

#active_object_mask_updated = [] #not used by repo


img_first_box_updated = []
img_last_box_updated = []
img_first_rel_updated = []
img_last_rel_updated = []

boxes_512_filtered = []
boxes_1024_filtered = []
attributes_filtered = []
labels_filtered = []

preds_filtered = []
rels_filtered = []

#split_filtered = [] #stays the same

def masklist(mylist,mymask):
    return [a for a,b in zip(mylist,mymask) if b]

added_boxes = 0
added_rels = 0
del_boxes = 0
numrel_added = []

for i in range(len(f['img_to_first_box'])):
    box_indstart = f['img_to_first_box'][i]
    box_indend = f['img_to_last_box'][i]

    labels = f['labels'][box_indstart:box_indend]
    mask_labels = get_validinds(labels, ind_to_classes, 'bbox')

    

    boxes_512 = f['boxes_512'][box_indstart:box_indend]
    boxes_1024 = f['boxes_1024'][box_indstart:box_indend]
    attr = f['attributes'][box_indstart:box_indend]

    boxes_512_filtered.extend(masklist(boxes_512,mask_labels))
    boxes_1024_filtered.extend(masklist(boxes_512,mask_labels))
    attributes_filtered.extend(masklist(attr,mask_labels))
    labels_filtered.extend(masklist(labels,mask_labels))

    #del_box_inds.extend(masklist(labels,mask_labels))
    
    if mask_labels.count(True) > 0:
        img_first_box_updated.append(added_boxes)
        added_boxes = added_boxes + mask_labels.count(True)
        img_last_box_updated.append(added_boxes-1)
    else:
        img_first_box_updated.append(-1)
        img_last_box_updated.append(-1)        
    

    rel_indstart = f['img_to_first_rel'][i]
    rel_indend = f['img_to_last_rel'][i]

    rels = list(f['relationships'][rel_indstart:rel_indend])
    preds = list(f['predicates'][rel_indstart:rel_indend])
    removed_labels = [not i for i in mask_labels]

    #delete indices of the boxes in boxes array
    #print(mask_labels)
    del_ind = [] 
    for k, boollabel in enumerate(mask_labels):
        if boollabel==False:
            del_ind.append(box_indstart+k)
    
    #remove relationship which are unvalid
    removed_rel = 0
    for k,rel in enumerate(rels):
        if rel[0] in del_ind or rel[1] in del_ind:
            del rels[k]
            del preds[k]
            removed_rel = removed_rel + 1
            #print("removed rel ", k , rel)

    #account for invalid indeces in relationship array because of boxes removed
    #relative to current frame
    #print(np.max(f['relationships'][rel_indstart:rel_indend]), len(boxes_512))
    #print("del index: ",del_ind)
    #print("del boxes ", del_boxes)
    #rels = [rels[2]]
    #print("rel before: ",rels)
    subtract = [[0,0] for r in range(len(rels))]
    #print(subtract)
    for k in del_ind:
        for n,rel in enumerate(rels):
            if rel[0]>k:
                subtract[n][0] = subtract[n][0] + 1
                #print("-", rel)
            if rel[1]>k:
                subtract[n][1] = subtract[n][1] + 1
                #print("-",rel)
    #print(subtract)
    rels = list(np.array(rels) - np.array(subtract))
    """print("rel after: ",rels)
    print(len(boxes_512),len(boxes_512_filtered))
    print(boxes_512)
    print(boxes_512_filtered)"""

    #relative to all previous box removals, previous iterations
    for rel in rels:
        rel[0] = rel[0] - del_boxes
        rel[1] = rel[1] - del_boxes
    del_boxes = del_boxes + len(del_ind)

    #mask_preds = get_validinds(preds, ind_to_classes, 'rel')
    rels_filtered.extend(rels)
    preds_filtered.extend(preds)
    numrel_added.append(len(rels))


    if len(rels) > 0:
        img_first_rel_updated.append(added_rels)
        added_rels = added_rels + len(rels)
        img_last_rel_updated.append(added_rels-1)
    else:
        img_first_rel_updated.append(-1)
        img_last_rel_updated.append(-1)

    #boxes_512_filtered = np.array(boxes_512_filtered)
    #labels_filtered = np.array(labels_filtered)
    #rels_filtered = np.array(rels_filtered)
    """print("f['boxes_512']: ",f['boxes_512'][box_indstart:box_indend])
    print('boxes_512_filtered: ', boxes_512_filtered)
    print("f['relationships']: ",f['relationships'][rel_indstart:rel_indend])
    print('rels_filtered: ', rels_filtered)

    print(len(boxes_512), len(boxes_512_filtered))
    for rel in rels_filtered:
            print(boxes_512_filtered[rel[0]], boxes_512_filtered[rel[1]])
    exit(1)"""
    


    if i%1000 == 0 and i!=0:
        print("Processed %d images."%i)
    """print("i for loop",i)"""
    if i==1:
        for j in range(i):
            print("added: ",numrel_added)
            i_rel_start = img_first_rel_updated[j]
            i_rel_end = img_last_rel_updated[j]
            print("number rels: ", i_rel_end- i_rel_start + 1)
            print(i_rel_start, i_rel_end)
            i_obj_start = img_first_box_updated[j]
            i_obj_end = img_last_box_updated[j]
            print(i_obj_start, i_obj_end)

            boxes_i = np.array(boxes_512_filtered[i_obj_start : i_obj_end + 1])
            i_obj_start = img_first_box_updated[j]

            print(rels_filtered[i_rel_start : i_rel_end + 1])
            print(i_obj_start) # range is [0, num_box)
            #print("obj_idx: ",obj_idx)
            print("shape: ",boxes_i.shape)

            print("gt:")
            i_rel_start = f['img_to_first_rel'][j]
            i_rel_end = f['img_to_last_rel'][j]
            print("number rels: ", i_rel_end- i_rel_start + 1)
            print(i_rel_start, i_rel_end)
            i_obj_start = f['img_to_first_box'][j]
            i_obj_end = f['img_to_last_box'][j]
            print(i_obj_start, i_obj_end)
            print(f['relationships'][i_rel_start:i_rel_end+1])
            print(f['boxes_512'][i_obj_start : i_obj_end + 1].shape)

        break
    
    """if i==2:
        

        img_first_box_updated = np.array(img_first_box_updated)
        img_last_box_updated = np.array(img_last_box_updated)
        img_first_rel_updated = np.array(img_first_rel_updated)

        img_last_rel_updated = np.array(img_last_rel_updated)
        boxes_512_filtered = np.array(boxes_512_filtered)
        labels_filtered = np.array(labels_filtered)
        rels_filtered = np.array(rels_filtered)
        attributes_filtered = np.array(attributes_filtered)
        
        print("f['img_to_first_box']: ",f['img_to_first_box'][:3])
        print("f['img_to_last_box']: ",f['img_to_first_box'][:3])
        print("f['img_to_first_rel']: ", f['img_to_first_rel'][:3])
        print("f['img_to_last_rel']: ", f['img_to_last_rel'][:3])
        print("f['boxes_512']: ",f['boxes_512'][:40])
        print('boxes_512_filtered: ', boxes_512_filtered)
        print("f['relationships']: ",f['relationships'])
        print('rels_filtered: ', rels_filtered)
        exit(1)
        print("f['predicates']: ",f['predicates'])

        print("%s"%['*']*40)

        print('img_first_box_updated: ',img_first_box_updated, type(img_first_box_updated[0]))
        print('img_last_box_updated: ', img_last_box_updated, type(img_last_box_updated[0]))
        print('img_first_rel_updated: ',img_first_rel_updated, type(img_first_rel_updated),type(img_first_rel_updated[0]))
        print('img_last_rel_updated: ', img_last_rel_updated)

        print('boxes_512_filtered: ', boxes_512_filtered)
        #print('boxes_1024_filtered: ', boxes_1024_filtered)

        #print('attributes_filtered: ', attributes_filtered)
        print('labels_filtered: ', labels_filtered)
        print('preds_filtered: ', preds_filtered)
        print('rels_filtered: ', rels_filtered)

        for rel in rels_filtered:
            print(boxes_512_filtered[rel[0]], boxes_512_filtered[rel[1]])
        print()
        for rel in f['relationships'][:40]:
            print(f['boxes_512'][rel[0]], f['boxes_512'][rel[1]])

        break
        """

#Save updated datasets to file
filename = "%s-subset.h5"%os.path.splitext(os.path.basename(args.file))[0]
data_file = h5py.File(os.path.join(args.outputdir, filename), 'w')
data_file.create_dataset('active_object_mask', data=f['active_object_mask'])
data_file.create_dataset('attributes', data=attributes_filtered)
data_file.create_dataset('boxes_1024', data=boxes_1024_filtered)
data_file.create_dataset('boxes_512', data=boxes_512_filtered)
data_file.create_dataset('img_to_first_box', data=img_first_box_updated)
data_file.create_dataset('img_to_last_box', data=img_last_box_updated)
data_file.create_dataset('img_to_first_rel', data=img_first_rel_updated)
data_file.create_dataset('img_to_last_rel', data=img_last_rel_updated)
data_file.create_dataset('labels', data=labels_filtered)
data_file.create_dataset('predicates', data=preds_filtered)
data_file.create_dataset('relationships', data=rels_filtered)
data_file.create_dataset('split', data=f['split'])

data_file.close()