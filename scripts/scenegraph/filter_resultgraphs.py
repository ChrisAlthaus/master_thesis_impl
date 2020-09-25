import argparse
import os

import datetime
import json
import random
import time

import itertools
import numpy as np

from validlabels import ind_to_classes, ind_to_predicates, VALID_BBOXLABELS, VALID_RELLABELS

def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-file', help='Path to the prediction graph file.')
    parser.add_argument('-imginfo', help='Path to the image info file.')
    parser.add_argument('-outputdir')
    parser.add_argument('-boxestresh', type=float, default=0.01)
    parser.add_argument('-relstresh', type=float, default=0.01)
    parser.add_argument('-build_labelvectors', action='store_true')
    parser.add_argument('-relsasnodes', action='store_true')

    args = parser.parse_args()
    return args

def main(args):
    if not os.path.isfile(args.file) or not os.path.isfile(args.imginfo):
        raise ValueError("Json file does not exist.")
    if not os.path.isdir(args.outputdir):
        raise ValueError("Output directory does not exist.")

    output_dir = os.path.join(args.outputdir, datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        raise ValueError("Output directory %s already exists."%output_dir)

    data = None
    with open(args.file, "r") as f:
        data = json.load(f)

    imginfo = None
    with open(args.imginfo, "r") as f:
        imginfo = json.load(f)

    #Graph file format:
    #   every dict item represents one image prediction
    #       ->fields are:   bbox (80, 4)    (sizes may differ)
    #                       bbox_labels (80,)
    #                       bbox_scores (80,)
    #                       rel_pairs (6320, 2)
    #                       rel_labels (6320,)
    #                       rel_scores (6320,)
    #                       rel_all_scores (6320, 51)

    idx_to_files = imginfo["idx_to_files"]
    ind_to_classes = imginfo["ind_to_classes"]
    ind_to_predicates = imginfo["ind_to_predicates"]
    _SCORE_THRESH_BOXES = args.boxestresh
    _SCORE_THRESH_RELS = args.relstresh
    _RELS_AS_NODES = args.relsasnodes

    _BOXES_TOPK = 20
    _RELS_TOPK = 20
    graphs = dict()
    ann = ''

    b_labels_added = []
    r_labels_added = []

    for idx, preds in list(data.items()):
        box_labels = preds['bbox_labels']   #assumption: model just supports valid labels
        box_scores = preds['bbox_scores']   

        #filter out unvalid boxes
        b_valid = []
        b_scores = []
        b_indices = [] 
        b_classes = []
        skipped_indices = []
        for i, l in enumerate(box_labels):
            if ind_to_classes[l] in VALID_BBOXLABELS and box_scores[i]>=_SCORE_THRESH_BOXES:
                b_valid.append(l)
                b_indices.append(i)
                b_scores.append(box_scores[i])
                b_classes.append(ind_to_classes[l])
                #Logging
                ann = ann + str(i) + '_' + ind_to_classes[l] +'; '+ str(box_scores[i]) + '\n'
                #Logging end
            else:
                skipped_indices.append(i)
            if len(b_valid) >= _BOXES_TOPK:
                break
            
        print("Number of valid boxes: ",b_valid)
        rels = preds['rel_pairs']
        rel_scores = preds['rel_scores']
        rel_labels = preds['rel_labels']

        """#delete all rels with skipped box correspondance
        for i in reversed(list(range(len(rels)))):
            if rels[i][0] in skipped_indices or rels[i][1] in skipped_indices:
                del rels[i]
                del rel_scores[i]
                del rel_labels[i]"""

        #filter out unvalid relations
        r_valid = []
        r_scores = []
        r_labels = []

        for i,rel in enumerate(rels):
            if rel[0] in b_indices and rel[1] in b_indices:
                if rel_scores[i] >= _SCORE_THRESH_RELS:
                    r_valid.append(rel)
                    r_scores.append(rel_scores[i])
                    r_labels.append(rel_labels[i])

                    #Logging
                    b1_label = ind_to_classes[ b_valid[ b_indices.index(rel[0]) ] ]
                    b2_label = ind_to_classes[ b_valid[ b_indices.index(rel[1]) ] ]
                    relstr = str(rel[0]) + '_' + b1_label + ' => ' + ind_to_predicates[rel_labels[i]] + ' => ' + str(rel[1]) + '_' + b2_label
                    relstr = relstr + '; '+ str(rel_scores[i])
                    ann = ann + relstr + '\n'
                    #Logging end
            if len(r_valid)>_RELS_TOPK:
                break
            

        r_classes = [ind_to_predicates[l] for l in r_labels]

        """#align rels indices to match the new box ordering
        print("r_valid: ",r_valid)
        for i in skipped_indices:
            for rel in r_valid:
                if rel[0] > i:
                    rel[0] = rel[0] - 1
                if rel[1] > i:
                    rel[1] = rel[1] - 1

        print("skipped: ",skipped_indices)
        print("r_valid: ",r_valid)"""


        b_labels_added.append(b_valid)
        r_labels_added.append(r_labels)
        print("Annotations filtered (for validation purposes): \n", ann)

        r_indices = range(max(b_valid), max(b_valid) + len(r_valid))
        if _RELS_AS_NODES:  #e.g. triple person - before - mountain -> edges (not labeled!): {(person, before), (before,mountain)}
                            #  number of edges = 2*num old edges
            r_valid_plus = []
            for i,rel in enumerate(r_valid):
                relsrc = [rel[0], r_indices[i]]
                reldst = [r_indices[i], rel[1]]
                r_valid_plus.extend([relsrc, reldst])

            graph = {'edges': r_valid_plus, 'features': {ind:c_label for ind,c_label in list(zip(b_indices,b_classes))}}
            graph['features'].update({ind:r_label for ind,r_label in list(zip(r_indices,r_classes))})
            graph.update({'box_scores': {ind:b_score for ind,b_score in list(zip(b_indices,b_scores))}})
            graph.update({'rel_scores': {ind:r_score for ind,r_score in list(zip(r_indices,r_scores))}})

        else:    #e.g. triple person - before - mountain -> edges (not labeled!): {(person,mountain)}
            graph = {'edges': r_valid, 'features': {ind:c_label for ind,c_label in list(zip(b_indices,b_classes))}}
            graph.update()
            graph.update({'box_scores': {ind:b_score for ind,b_score in list(zip(b_indices,b_scores))}})
            graph.update({'rel_scores': r_scores})

        graphs[idx_to_files[int(idx)]] = graph

    if len(data) == 1:
        annname = "labels.txt"
        with open(os.path.join(output_dir, annname), "w") as text_file:
            text_file.write(ann)


    print("Number of graphs in output file: ",len(graphs))
    outfile_name = "graphs-topk.json"
    with open(os.path.join(output_dir, outfile_name), 'w') as f:
        print("Writing to file: ",os.path.join(output_dir, outfile_name))
        json.dump(graphs, f)   

    configstr = 'Box Treshold: %f \n'%_SCORE_THRESH_BOXES
    configstr = configstr + 'Rel Treshold: %f \n'%_SCORE_THRESH_RELS
    configstr = configstr + 'Rels as Nodes: %s \n'%str(_RELS_AS_NODES)
    with open(os.path.join(output_dir, 'config.txt'), 'w') as text_file:
        text_file.write(configstr)

    if args.build_labelvectors:
        labelvectors = []
        for idx, preds in enumerate(list(zip(b_labels_added, r_labels_added))):
            box_labelvector, rels_labelvector = getlabelvectors(preds[0], preds[1], _BOXES_TOPK, _RELS_TOPK)
        
            labelvectors.append({'imagefile': idx_to_files[int(idx)], 'boxvector': box_labelvector.tolist(), 'relvector': rels_labelvector.tolist()})

        print("Number of labelvectors in output file: ",len(data))
        outfile_name = "labelvectors-topk.json"
        with open(os.path.join(output_dir, outfile_name), 'w') as f:
            print("Writing to file: ",os.path.join(output_dir, outfile_name))
            json.dump(labelvectors, f) 


def getlabelvectors(box_labels, rel_labels, boxes_topk, rels_topk):
    from validlabels import ind_to_classes, ind_to_predicates, VALID_BBOXLABELS, VALID_RELLABELS

    if isinstance(box_labels[0],str):
        box_labels = [ind_to_classes.index(l) for l in box_labels]
    if len(rel_labels) > 0:
        if isinstance(rel_labels[0],str):
            rel_labels = [ind_to_predicates.index(l) for l in rel_labels]
    
    b_labels = []
    for l in box_labels:
        #sanity check for valid labels
        if ind_to_classes[l] in VALID_BBOXLABELS:
            b_labels.append(l)
        if len(b_labels) >= boxes_topk and boxes_topk != -1:
            break
    
    r_labels = []
    for r in rel_labels:
        #sanity check for valid labels
        if ind_to_predicates[r] in VALID_RELLABELS:
            r_labels.append(r)
        if len(r_labels) >= rels_topk and rels_topk != -1:
            break
 
    box_labelvector = np.zeros(len(VALID_BBOXLABELS))
    for l in b_labels:
        b_class = ind_to_classes[l]
        box_labelvector[VALID_BBOXLABELS.index(b_class)] = box_labelvector[VALID_BBOXLABELS.index(b_class)] + 1

    rels_labelvector = np.zeros(len(VALID_RELLABELS))
    for r in r_labels:
        r_class = ind_to_predicates[r]
        rels_labelvector[VALID_RELLABELS.index(r_class)] = rels_labelvector[VALID_RELLABELS.index(r_class)] + 1

    return box_labelvector, rels_labelvector

 

if __name__ == "__main__":
    args = parameter_parser()
    main(args)



