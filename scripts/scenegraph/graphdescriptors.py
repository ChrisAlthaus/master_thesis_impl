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
    parser.add_argument('-boxestopk', type=int, default=20)
    parser.add_argument('-relstopk', type=int, default=20)
    parser.add_argument('-build_labelvectors', action='store_true')
    parser.add_argument('-relsasnodes', action='store_true')

    args = parser.parse_args()
    return args

#Transform the input graph prediction file (bbox, bbox_labels,...) to the necessary 
#Graph2Vec file format (edgeindices, boxlabels, ...). 
#Previously filter the graph predictions by:
#   1.bbox score treshold
#   2.valid bbox label
#   3.take topk of these filtered
#Previously filter the relationship predictions by:
#   1.rel score treshold
#   2.valid rel label
#   3.take topk of these filtered
#Additional: Save distribution of bbox and relationship labels of filtered results
#(labelsvectors-topk.json, used for weight reranking of search results later)


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

    #File formats:
    #Graph file format (input):
    #   every dict item represents one image prediction
    #       ->fields are:   bbox (80, 4)    (sizes may differ)
    #                       bbox_labels (80,)
    #                       bbox_values (80,)
    #                       rel_pairs (6320, 2)
    #                       rel_labels (6320,)
    #                       rel_values (6320,)
    #                       rel_all_values (6320, 51)
    #
    #Graph2Vec file format (output):
    #   dict with entries:
    #           -imgpath
    #           -edges (indices pairs, refering to feature field)
    #           -features: dict mapping ind->bbox/rel label
    #           -box_values: dict mapping ind->bbox score
    #           -rel_values: dict mapping ind->rel score

    idx_to_files = imginfo["idx_to_files"]
    ind_to_classes = imginfo["ind_to_classes"]
    ind_to_predicates = imginfo["ind_to_predicates"]
    _SCORE_THRESH_BOXES = args.boxestresh
    _SCORE_THRESH_RELS = args.relstresh
    _RELS_AS_NODES = args.relsasnodes

    _BOXES_TOPK = args.boxestopk
    _RELS_TOPK = args.relstopk
    graphs = dict()
    ann = ''

    c_noann = 0

    b_labels_added = []
    r_labels_added = []

    c_boxes = []
    c_rels = []

    for idx, preds in list(data.items()):

        box_labels = preds['bbox_labels']   #assumption: model just supports valid labels
        box_values = preds['bbox_values']
        rels = preds['rel_pairs']
        rel_values = preds['rel_values']
        rel_labels = preds['rel_labels']

        if len(rels)<= 0 or len(box_labels)<= 1:
            c_noann = c_noann + 1
            continue

        b_valid = preds['bbox_labels'] 
        b_values = preds['bbox_values']
        b_indices = list(range(0,len(b_valid)))
        b_classes = list(map(lambda x: ind_to_classes[x], b_valid))
        r_valid = preds['rel_pairs']
        r_values = preds['rel_values']
        r_labels = preds['rel_labels']
        r_classes =  list(map(lambda x: ind_to_predicates[x], r_labels))


        b_labels_added.append(b_valid)
        r_labels_added.append(r_labels)
        c_boxes.append(len(b_valid))
        c_rels.append(len(r_valid))
        #print("Annotations filtered (for validation purposes): \n", ann)

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
            graph.update({'box_values': {ind:b_score for ind,b_score in list(zip(b_indices,b_values))}})
            graph.update({'rel_values': {ind:r_score for ind,r_score in list(zip(r_indices,r_values))}})

        else:    #e.g. triple person - before - mountain -> edges (not labeled!): {(person,mountain)}
            graph = {'edges': r_valid, 'features': {ind:c_label for ind,c_label in list(zip(b_indices,b_classes))}}
            graph.update()
            graph.update({'box_values': {ind:b_score for ind,b_score in list(zip(b_indices,b_values))}})
            graph.update({'rel_values': r_values})

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
    
    configstr = ''
    #configstr = 'Box Treshold: %f \n'%_SCORE_THRESH_BOXES
    #configstr = configstr + 'Rel Treshold: %f \n'%_SCORE_THRESH_RELS
    configstr = configstr + 'Rels as Nodes: %s \n'%str(_RELS_AS_NODES)
    configstr = configstr + 'Number of input predictions: %d \n'%len(data)
    configstr = configstr + 'Number of generated graph descriptors: %d \n'%len(graphs)
    configstr = configstr + 'Predictions with either no rels or no bboxes: %d \n'%c_noann
    configstr = configstr + 'Distribution of number of bboxes: {} \n'.format(getwhiskersvalues(c_boxes))
    configstr = configstr + 'Distribution of number of rels: {} \n'.format(getwhiskersvalues(c_rels))


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


def getwhiskersvalues(values):
    #Computes the box-plot whiskers values.
    #Computed values: min, low_whiskers, Q1, median, Q3, high_whiskers, max
    #Ordering differs from whiskers plot ordering.
    Q1, median, Q3 = np.percentile(np.asarray(values), [25, 50, 75])
    IQR = Q3 - Q1

    loval = Q1 - 1.5 * IQR
    hival = Q3 + 1.5 * IQR

    wiskhi = np.compress(values <= hival, values)
    wisklo = np.compress(values >= loval, values)
    actual_hival = np.max(wiskhi)
    actual_loval = np.min(wisklo)

    Qs = [actual_loval, loval, Q1, median, Q3, hival, actual_hival]
    Qname = ["Actual LO", "Q1-1.5xIQR", "Q1", "median", "Q3", "Q3+1.5xIQR", 
             "Actual HI"]
    logstr = ''.join(["{}:{} ".format(a,b) for a,b in zip(Qname,Qs)])
    return logstr

if __name__ == "__main__":
    args = parameter_parser()
    main(args)



