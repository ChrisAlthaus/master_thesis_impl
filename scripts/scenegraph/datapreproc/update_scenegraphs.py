import h5py

import argparse
import os
import json

import numpy as np

from validlabels import VALID_BBOXLABELS

parser = argparse.ArgumentParser()
parser.add_argument('-file', help='Path to the scene graph file.') #VG-SGG-with-attri.h5
parser.add_argument('-trainandtest', action='store_true')
parser.add_argument('-outputdir')

args = parser.parse_args()

f = h5py.File(args.file, 'r')

#Filter out all not wanted bbox labels & write the result to a new .h5 file.
#Predicates are not filtered, but relationships have to be filtered also.
#With args.trainandset can be specified if all annotations (train+ validation set)
#or just the validation set should be transformed.
#Fields of input .h5 file:
# - 'img_to_first_box': Update indices based on removed boxes
# - 'img_to_last_box': Update indices based on removed boxes
# - 'img_to_first_rel': Update indices based on removed relationships
# - 'img_to_last_rel': Update indices based on removed relationships 
# - 'labels': calculate intersection with valid labels
# - 'boxes_1024': reduce with intersect of valid labels
# - 'boxes_512': reduce with intersect of valid labels
# - 'attributes': reduce with intersect of valid labels
# - 'relationships': Containing indices of boxes_512/1024. 
#                    Removed & updated based on filtered boxes (tedious, because refering to removed bbox indices)
# - 'predicates': reduces with reference to relationships
# - 'active_object_mask': not edited
# - 'split': 2 for validation or 0 for train set
#
#Validation of results possible with script: scenegraph/visualizedata.py


def get_validinds(labels, ind_to_classes, type):
    valid_inds = [ind_to_classes.index(elem) for elem in VALID_BBOXLABELS]
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
    # ---------------------------- Update Box indices & data ----------------------------
    box_indstart = f['img_to_first_box'][i]
    box_indend = f['img_to_last_box'][i]

    labels = f['labels'][box_indstart:box_indend+1]
    mask_labels = get_validinds(labels, ind_to_classes, 'bbox')

    boxes_512 = f['boxes_512'][box_indstart:box_indend+1]
    boxes_1024 = f['boxes_1024'][box_indstart:box_indend+1]
    attr = f['attributes'][box_indstart:box_indend+1]

    if f['split'][i] == 2 or args.trainandtest: 
        boxes_512_filtered.extend(masklist(boxes_512,mask_labels))
        boxes_1024_filtered.extend(masklist(boxes_1024,mask_labels))
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
    else:
        boxes_512_filtered.extend(boxes_512)
        boxes_1024_filtered.extend(boxes_1024)
        attributes_filtered.extend(attr)
        labels_filtered.extend(labels)

        if len(boxes_512)>0:
            img_first_box_updated.append(added_boxes)
            added_boxes = added_boxes + len(boxes_512)
            img_last_box_updated.append(added_boxes-1)
        else:
            img_first_box_updated.append(-1)
            img_last_box_updated.append(-1) 

    # ---------------------------- Update Relationship indices & data ----------------------------
    rel_indstart = f['img_to_first_rel'][i]
    rel_indend = f['img_to_last_rel'][i]

    rels = list(f['relationships'][rel_indstart:rel_indend+1])
    preds = list(f['predicates'][rel_indstart:rel_indend+1])

    if f['split'][i] == 2 or args.trainandtest:
        removed_labels = [not i for i in mask_labels]

        #delete indices of the boxes in boxes array
        del_ind = [] 
        for k, boollabel in enumerate(mask_labels):
            if boollabel==False:
                del_ind.append(box_indstart+k)

        #remove relationship which are unvalid
        removed_rel = 0
        for k in list( reversed(range(len(rels))) ):
            rel = rels[k]
            if rel[0] in del_ind or rel[1] in del_ind:
                del rels[k]
                del preds[k]
                removed_rel = removed_rel + 1

        #sanity check
        for rel in rels:
            assert any((rel == x).all() for x in list(f['relationships'][rel_indstart:rel_indend+1]))


        #account for invalid indeces in relationship array because of boxes removed
        #relative to current frame
        subtract = [[0,0] for r in range(len(rels))]
        for k in del_ind:
            for n,rel in enumerate(rels):
                if rel[0]>=k:
                    subtract[n][0] = subtract[n][0] + 1
                if rel[1]>=k:
                    subtract[n][1] = subtract[n][1] + 1          
        rels = list(np.array(rels) - np.array(subtract))

        #relative to all previous box removals, previous iterations
        for rel in rels:
            rel[0] = rel[0] - del_boxes
            rel[1] = rel[1] - del_boxes
        del_boxes = del_boxes + len(del_ind)

        #sanity check
        num_boxes = len(masklist(boxes_512,mask_labels))
        for rel in rels:
            assert (rel<added_boxes).all(), print(rels,added_boxes)

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
    else:
        #relative to all previous box removals, previous iterations
        for rel in rels:
            rel[0] = rel[0] - del_boxes
            rel[1] = rel[1] - del_boxes

        rels_filtered.extend(rels)
        preds_filtered.extend(preds)

        if len(rels) > 0:
            img_first_rel_updated.append(added_rels)
            added_rels = added_rels + len(rels)
            img_last_rel_updated.append(added_rels-1)
        else:
            img_first_rel_updated.append(-1)
            img_last_rel_updated.append(-1)

    
    if i%1000 == 0 and i!=0:
        print("Processed %d images."%i)

#Save updated datasets to file
filename = "%s-subset-%s.h5"%(os.path.splitext(os.path.basename(args.file))[0], 'trainval' if args.trainandtest else 'val')
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