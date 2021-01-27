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
from gensim.test.utils import get_tmpfile
import dill


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

graph2vec_dir = '/home/althausc/master_thesis_impl/graph2vec/src'
sys.path.insert(0,graph2vec_dir) 
sgraphscript_dir = '/home/althausc/master_thesis_impl/scripts/scenegraph'
sys.path.insert(0,sgraphscript_dir) 

from graph2vec import WeisfeilerLehmanMachine, dataset_reader, feature_extractor, save_embedding

def main():
    parser = argparse.ArgumentParser(description="Run Graph2Vec.")
    parser.add_argument("--model", help="Path to the previous trained Doc2Vec model.")
    parser.add_argument("--inputpath", help="Input folder with jsons.")
    #parser.add_argument("--modelout", help="Path to the directory where the model should be saved.") #deprecated?
    parser.add_argument("--inference", action='store_true',
                        help="Using stored model for inference on a given graph.")
    parser.add_argument("--reweight", action='store_true',
                        help="Include the box & rel labels to reweight the top k results.")
    parser.add_argument("--reweightmode", default='jaccard', help="Mode for similarity calculations between labelvectors.")
    parser.add_argument("--labelvecpath", help="Path to the g2v training set composed of all graphs.")
    parser.add_argument("--topk", type=int, default=100,
                        help="Number of returned similar documents.") 
    parser.add_argument('-metadata_imgpath', help='To provide name of search input image for resultdirectories configfile.')
    parser.add_argument("--wl-iterations", type=int, default=2,
    	                help="Number of Weisfeiler-Lehman iterations. Default is 2.") 
    parser.add_argument("--steps-infer", type=int, default=100,
    	                help="Number of steps for the doc2vec inference function.")
    parser.add_argument("--min-featuredim", type=int, default=136, 
    	                help="Extend feature by itself when too small. Mean feature length of train dataset may be a good choice (disable: -1).")                        
    args = parser.parse_args()

    #Given an input graph in the graph2vec-format search for the topk most similar graphs.
    #For these return the corresponding ranked images (with scores) 

    _REWEIGHT_MODES = ['euclid', 'jaccard']

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
    for k,d in enumerate(document_collections):
        print("Feature length of document {} = {}".format(k, len(d.words)))
        if args.min_featuredim == -1:
            break

        basefeatures = copy.deepcopy(d.words)
        add = False
        while len(d.words)<= args.min_featuredim:
            d.words.extend(basefeatures)
            add = True
        if add:
            print("Extended feature from length {} to length {}".format(len(basefeatures), len(d.words)))

    # ------------------------ Get most similar topk graphs with corresponding image ---------------------
    if args.inference:
        with open(args.model,'rb') as f:
            print("Load model: ", args.model) 
            model = dill.load(f)

        output_dir = os.path.join('/home/althausc/master_thesis_impl/retrieval/out/scenegraphs', datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            raise ValueError("Output directory %s already exists."%output_dir)

        print("Doc2Vec state: ")
        print(', '.join("%s: %s" % item for item in vars(model).items()))
        print("Features of documents: ",document_collections)
        print(type(document_collections))

        if len(document_collections)<= 0:
            saveresults([], output_dir)
            return

        #Subsequent calls to the inference function may infer different representations for the same document. 
        #For a more stable representation, increase the number of steps to assert a stricket convergence.  
        #model.random.seed(0)
        vector = model.infer_vector(document_collections[0].words, steps=args.steps_infer) #TODO: do eval of best step size, Only 1 graph?? TODO:batch of graphs
        sims = model.docvecs.most_similar([vector], topn = args.topk)
        print("Top k similarities (with cos-sim): ",sims)

        # -------------------- Reweight based on query & results label distributions -----------------------
        if args.reweight: #reweight on box and rel labels occurance similarities on the query and the result topk
                          #e.g. annotation a1 with ['tree' * 5, ...] should be ranked higher for query image including ['tree']
                          #     than for example a2 with ['car' *2, ...]    #NOTE: not build-in & tested so far!!!
            if args.reweightmode not in _REWEIGHT_MODES:
                raise ValueError("No valid reweight mode specified.")
            
            labeldata = None
            with open(args.labelvecpath, "r") as f:
                labeldata = json.load(f)

            from filter_resultgraphs import getlabelvectors
            #print(graphs) 
            graph = graphs[0][1] 
            print("Query image graph: ",graph)
            g_features = sorted(list(graph['features'].items()), key=lambda x: int(x[0]))
            print(graph['features'].items())
            g_features = [x[1] for x in g_features]
            print("g_features: ",g_features)
            g_bclasses = g_features[:len(graph['box_scores'])]
            print("length box_scores: ",len(graph['box_scores']))
            g_rclasses = g_features[len(graph['box_scores']): len(graph['box_scores']) + len(graph['rel_scores'])]
            print("len rel scores: ",len(graph['rel_scores']))
            print(g_bclasses)
            print(g_rclasses)

            g_boxlvec, g_rellvec = getlabelvectors(g_bclasses, g_rclasses, -1, -1)
            print("g_rellvec: ",g_rellvec)
            g_labelvec = np.concatenate((g_boxlvec, g_rellvec))    #TODO: resume
            print("g_labelvec: ",g_labelvec)
            #g_labelvec = np.stack([g_boxlvec, g_rellvec]).reshape(-1)

            reweight_scores = []
            imgfiles = []

            for imgfile, score in sims:
                for item in labeldata:
                    if os.path.basename(item['imagefile']) == os.path.basename(imgfile):
                        #print("item: ", item)
                        labelvec = item['boxvector']    #length of boxvector = number of valid classes
                        labelvec.extend(item['relvector']) #length of relvector = number of valid relationship predicates


                        #print("g_labelvec: ",g_labelvec)
                        #print("labelvec: ",labelvec)
                        labelsim = None
                        if args.reweightmode == _REWEIGHT_MODES[0]:
                            labelsim = np.linalg.norm(g_labelvec - labelvec)
                        elif args.reweightmode == _REWEIGHT_MODES[1]:
                            metadata = [imgfile]
                            labelsim = getjaccard(g_labelvec, labelvec, metadata)

                        imgfiles.append(imgfile)
                        reweight_scores.append(labelsim)
                        break

            #normalize reweight vector to other range
            rs = np.array(reweight_scores)
            print("Reweight score before normalization: ",rs)
            scalemin, scalemax = 0.5, 1 #TODO: as arguments?
            rs = ( (rs - np.min(rs)) / (np.max(rs) - np.min(rs)) ) * (scalemax - scalemin) + scalemin
    
            print("Image names: ",imgfiles)
            print("Reweight scores normalized: ", rs)
            #reweight topk results which are found in train labelvectors
            notfound = 0
            print(sims)
            for i in range(len(sims)):
                if sims[i][0] in imgfiles:
                    r_ind = imgfiles.index(sims[i][0])
                    sims[i] = [sims[i][0], sims[i][1]*rs[r_ind]]
                else:
                    notfound = notfound + 1
            print("Number of not reweighted retrieval results: %d"%notfound)
            sims.sort(key=lambda x: x[1], reverse=True)

            print("Reweighted topk images: ",sims)

        # -------------------------- Save resulting ranking ---------------------------
        saveconfigsearch(output_dir, args, len(sims))
        saveresults(sims, output_dir)

    # ------------------------- Evaluate model w.r. to infer parameter step size ---------------------
    elif args.evaluations:
        #Specify the right step size parameter for inference
        #Querying with same images & specify similarities between the best-matching which should be the input image
        fname = get_tmpfile(args.model)
        model = Doc2Vec.load(fname)

        num_docs = model.corpus_count
        dim = model.vector_size

        steps = np.arange(50, 1000, 50)
        evalscores = []
        for s in steps:
            evalscore = 0
            for doc in document_collections:
                imgname = doc.tags[0]
                #model.random.seed(0)
                vector = model.infer_vector(doc.words, steps=s) #TODO: do eval of best step size
                sims = model.docvecs.most_similar([vector], topn = num_docs)

                for item in sims:
                    if item[0] == imgname:
                        evalscore = evalscore + item[1]
                        break
            evalscores.append(evalscore)

        #Write files to same directory as model
        output_dir = os.path.dirname(args.model)
        df = pd.DataFrame({'steps':steps , 'evaluation score':evalscores})
        ax = sns.relplot(x="steps", y='evaluation score', sort=False, kind="line", markers=True, data=df)
        ax.fig.savefig(os.path.join(output_dir,"eval_g2vmodel_c%dd_%d.png"%(num_docs, dim)))
        plt.clf()

def getjaccard(vec1, vec2, metadata):
    #[1,0,0,0,1,0,2,3,0,0]
    labelvec1 = []
    for i,x in enumerate(vec1):
        labelvec1.extend([i]*int(x))
    labelvec2 = []
    for i,x in enumerate(vec2):
        labelvec2.extend([i]*int(x)) 

    #print("labelvec1: ",labelvec1) 
    #print("labelvec2: ",labelvec2) 
    s1, s2 = set(labelvec1), set(labelvec2) #TODO: Maybe with number of occurances included better?
    #print("s1: ",s1)
    #print("s2: ",s2)
    #logging
    from validlabels import VALID_BBOXLABELS, VALID_RELLABELS
    inters = s1.intersection(s2)
    print(metadata)
    print("Intersection classes: ", [VALID_BBOXLABELS[i] for i in inters if i< len(VALID_BBOXLABELS)])
    print("Intersection rels: ", [VALID_RELLABELS[i-len(VALID_BBOXLABELS)] for i in inters if i>= len(VALID_BBOXLABELS)])
    print("Number Intersections: ", len(s1.intersection(s2)))
    print("Number Union: ", len(s1.union(s2)))

    #logging end
    return len(s1.intersection(s2)) / len(s1.union(s2)) 

def saveconfigsearch(output_dir, args, numresults):
    with open(os.path.join(output_dir, 'config.txt'), 'a') as f:
        f.write("Model: %s"%args.model + os.linesep)
        f.write("Graph File: %s"%args.inputpath + os.linesep)
        f.write("Wl-Iterations: %d"%args.wl_iterations + os.linesep)
        f.write("Min Feature Dimension: %d"%args.min_featuredim + os.linesep)
        if args.reweight:
            f.write("Reweight Mode: %s"%args.reweightmode + os.linesep)
        f.write("Number of Results: %d"%numresults + os.linesep)
        if args.metadata_imgpath:
            f.write("Search Image: %d"%args.metadata_imgpath + os.linesep)

def saveresults(ranking, output_dir):
    #Output file format:
    #   - imagdir: base image directory
    #   - dict-entries rank : {filepath, relscore}
    outdata = {}
    prefixes = []
    for r,item in enumerate(ranking):
        fname = item[0]
        if fname.startswith('g_'):
            fname = fname[2:]
        prefixes.append(os.path.dirname(fname))
        fname = os.path.basename(fname)
        outdata[r] = {"filename": fname, "relscore": float(item[1])}
    
    assert all(x == prefixes[0] for x in prefixes)
    if len(ranking)==0:
        outdata['imagedir'] = 'not enough annotations'
    else:
        outdata['imagedir'] = prefixes[0]
    
    with open(os.path.join(output_dir, 'topkresults.json'), 'w') as f:
        print("Writing to file: ",os.path.join(output_dir, 'topkresults.json'))
        json.dump(outdata, f, indent=4, separators=(',', ': '))

if __name__ == "__main__":
    main()  







    

