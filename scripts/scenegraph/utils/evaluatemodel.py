#Path: /home/althausc/master_thesis_impl/graph2vec/src/evaluatemodel.py
import os,sys
import argparse
import json
import datetime
import numpy as np
import copy

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile

import glob
import hashlib
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import dill

graph2vec_dir = '/home/althausc/master_thesis_impl/graph2vec/src'
sys.path.insert(0,graph2vec_dir) 

from graph2vec import WeisfeilerLehmanMachine, dataset_reader, feature_extractor, save_embedding, getSimilarityScore

#Example Usage:
# python3.6 evaluatemodel.py --model /home/althausc/master_thesis_impl/graph2vec/models/12-11_17-10-51/g2vmodelc9211d128e20 
# --inputpath /home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/out/predictions/graphs/12-09_17-22-49/.descriptors/graphdescriptors.json 
# --wl-iterations 3 --min-featuredim -1

def main():
    parser = argparse.ArgumentParser(description="Run Graph2Vec.")
    parser.add_argument("--model", help="Path to the previous trained Doc2Vec model.")
    parser.add_argument("--inputpath", help="Input folder with jsons.")
    parser.add_argument("--wl-iterations", type=int, default=2,
    	                help="Number of Weisfeiler-Lehman iterations. Default is 2.") 
    parser.add_argument("--steps-infer", type=int, default=100,
    	                help="Number of steps for the doc2vec inference function.")
    parser.add_argument("--min-featuredim", type=int, default=136, 
    	                help="Extend feature by itself when too small. Mean feature length of train dataset may be a good choice (disable: -1).")                        
    args = parser.parse_args()

    data = None
    with open(args.inputpath, "r") as f:
        data = json.load(f)

    graphs = list(data.items()) #(filename, graph)
                                #graph = {'edges': ... , 'features': ... , 'box_scores': ... , 'rel_scores': ...}
    # ------------------------------ Feature Extraction of Input Graph(s) ---------------------------------
    print("\nFeature extraction started.\n")
    document_collections = Parallel(n_jobs=1)(delayed(feature_extractor)(gd[0], gd[1], args.wl_iterations) for gd in tqdm(graphs))
    print("\nOptimization started.\n")


     #Extend too small features if necessary with itself (because bad performace of small features)
    """for k,d in enumerate(document_collections):
        print("Feature length of document {} = {}".format(k, len(d.words)))
        if args.min_featuredim == -1:
            break

        basefeatures = copy.deepcopy(d.words)
        add = False
        while len(d.words)<= args.min_featuredim:
            d.words.extend(basefeatures)
            add = True
        if add:
            print("Extended feature from length {} to length {}".format(len(basefeatures), len(d.words)))"""

    print("Feature [0]: ", document_collections[0])

    with open(args.model,'rb') as f:
        model = dill.load(f)

    print("\nComputing evaluations started.\n")
    mscore, mrank, recalls = getSimilarityScore(document_collections, model, topk=100, stepsinfer=args.steps_infer)
    print("Mean Score: %f, Mean Rank: %s, Recalls: %s, Num documents: %d"%(mscore, str(mrank), str(recalls), len(document_collections)))

if __name__ == "__main__":
    main() 