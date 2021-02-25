#Path: /home/althausc/master_thesis_impl/graph2vec/src/graph2vec.py
"""Graph2Vec module."""

import json
import glob
import hashlib
import pandas as pd
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed
from param_parser import parameter_parser
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec
import os, sys
import datetime
import time
import shutil
import dill
import random
import csv
import copy
import matplotlib.pyplot as plt

sgraphscript_dir = '/home/althausc/master_thesis_impl/scripts/scenegraph'
sys.path.insert(0,sgraphscript_dir) 
sys.path.insert(0, '/home/althausc/master_thesis_impl/scripts/utils') 

from statsfunctions import getwhiskersvalues

class WeisfeilerLehmanMachine:
    """
    Weisfeiler Lehman feature extractor class.
    """
    def __init__(self, graph, features, iterations):
        """
        Initialization method which also executes feature extraction.
        :param graph: The Nx graph object.
        :param features: Feature hash table.
        :param iterations: Number of WL iterations.
        """
        self.iterations = iterations
        self.graph = graph
        self.features = features
        self.nodes = self.graph.nodes()
        self.extracted_features = [str(v) for k, v in features.items()]
        self.do_recursions()
        #print("Features: ",features)

    def do_a_recursion(self):
        """
        The method does a single WL recursion.
        :return new_features: The hash table with extracted WL features.
        """
        new_features = {}
        #print("Graph: ",self.graph)
        #print("Nodes: ",self.nodes)
        for node in self.nodes:
            #print("Node: ",node)
            nebs = list(self.graph.neighbors(node))
            #print("Neighbours: ",nebs)
            #print("f1= ",self.features)
            degs = [self.features[neb] for neb in nebs]
            #print("Degrees: ",degs)
            #Create sub-graph from current nodelabel including itself and all neighboring nodelabels
            #Subgraph is an unordered list of labels
            features = [str(self.features[node])]+sorted([str(deg) for deg in degs])
            #print("f= ",features)
            #Concatenating labels
            features = "_".join(features)
            #print("f= ",features)
            #Computing hash for each subgraph
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            #print("h= ",hashing)
            new_features[node] = hashing
            #exit(1)
        self.extracted_features = self.extracted_features + list(new_features.values())
        #print("Extracted: ",self.extracted_features)
        
        return new_features

    def do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for _ in range(self.iterations):
            #Doing feature extraction with reference to computed per-node features of last iteration (self.features)
            #Each iteration goes deeper into the graph
            self.features = self.do_a_recursion()


def dataset_reader(dataitem):   #modified
    """
    Function to read the graph and features from a json file.
    :param path: The path to the graph json.
    :return graph: The graph object.
    :return features: Features hash table.
    :return name: Name of the graph.
    """
    #print("graphdata: ",dataitem)
    #name = path.strip(".json").split("/")[-1]
    #data = json.load(open(path))
    graph = nx.from_edgelist(dataitem["edges"])

    if "features" in dataitem.keys():
        features = dataitem["features"]
    else:
        features = nx.degree(graph)

    features = {int(k): v for k, v in features.items()}
    #print("graph: ",dataitem["edges"])
    #print("features: ", features)
    #exit(1)
    return graph, features#, name

def feature_extractor(filename, graphdata, rounds):    #modified
    """
    Function to extract WL features from a graph.
    :param path: The path to the graph json.
    :param rounds: Number of WL iterations.
    :return doc: Document collection object.
    """
    graph, features = dataset_reader(graphdata)
    machine = WeisfeilerLehmanMachine(graph, features, rounds)
    doc = TaggedDocument(words=machine.extracted_features, tags=["g_" + filename])
    return doc

def save_embedding(output_path, model, files, dimensions):
    """
    Function to save the embedding.
    :param output_path: Path to the embedding csv.
    :param model: The embedding model object.
    :param files: The list of files.
    :param dimensions: The embedding dimension parameter.
    """
    out = []
    for f in files:
        #identifier = f.split("/")[-1].strip(".json")
        identifier = f
        out.append([identifier] + list(model.docvecs["g_"+identifier]))
    column_names = ["type"]+["x_"+str(dim) for dim in range(dimensions)]
    out = pd.DataFrame(out, columns=column_names)
    out = out.sort_values(["type"])
    out.to_csv(output_path, index=None)

    print("Wrote graph embedding to: ",os.path.dirname(output_path))

def save_prediction_emb(output_path, emb, dimensions):  #not used
    assert len(emb) == dimensions
    column_names = ["x_"+str(dim) for dim in range(dimensions)]
    out = pd.DataFrame([emb], columns=column_names)
    out.to_csv(output_path, index=None)

    print("Wrote graph embedding to file: ",output_path)



def getSimilarityScore(valdocs, model, topk, stepsinfer):
    #Querying with same images & specify similarities between the best-matching which should be the input image
    evalscore = 0
    numdocs_r20 = 0
    numdocs_r50 = 0
    numdocs_r100 = 0

    dim = model.vector_size

    modelcopy = copy.deepcopy(model)

    ranks = []
    start = time.time()

    for doc in valdocs:
        vector = model.infer_vector(doc.words, alpha=0.025, steps=stepsinfer)

    for doc in valdocs:
        imgname = doc.tags[0]
        #model.random.seed(0)
        #print(doc.words)
        #print(imgname)
        vector = modelcopy.infer_vector(doc.words, alpha=0.025, steps=stepsinfer)
        sims = modelcopy.docvecs.most_similar([vector], topn=topk)
        for r,item in enumerate(sims):
            #print(r,item)
            if item[0] == imgname:
                evalscore = evalscore + item[1]
                if r+1<=100:
                    numdocs_r100 = numdocs_r100 + 1
                if r+1<=50:
                    numdocs_r50 = numdocs_r50 + 1
                if r+1<=20:
                    numdocs_r20 = numdocs_r20 + 1

                ranks.append(r+1)
                break
    #print(ranks)

    end = time.time()
    #print("Validation took {} seconds".format(end-start))
    #print("Number of matched documents in top-%d = %d"%(num_docs, numdocs_found))
    #print("Mean rank number: ",sum(ranks)/len(ranks) if numdocs_found != 0 else 'Nan')
     
    mrank = sum(ranks)/len(ranks) if len(ranks) != 0 else 'Nan'
    mscore = evalscore/len(ranks)

    num_docs = len(valdocs)
    print(numdocs_r20, num_docs)
    print(numdocs_r50, num_docs)
    print(numdocs_r100, num_docs)
    print("-"*20)

    r20 = numdocs_r20/num_docs
    r50 = numdocs_r50/num_docs
    r100 = numdocs_r100/num_docs
    recalls = {'r@20': r20, 'r@50': r50, 'r@100': r100}
    return mscore, mrank, recalls
    
def getcallback(docs_collection, epochnum = 1, logdir=None, istrain=True, topk=100, stepsinfer=100):
    #Monitoring the loss value of every epoch
    class SimilarityCallback(CallbackAny2Vec):
        def __init__(self):
            self.epoch = 1
            self.docs = docs_collection
            self.epochnum = epochnum
            self.logdir = logdir
            self.csvfile = getlogforlosses(self.logdir, istrain, topk)
            self.stepsinfer = stepsinfer

        def on_epoch_end(self, model):
            if self.epoch % self.epochnum == 0:
                if istrain:
                    mscore, mrank, recalls = getSimilarityScore(self.docs[:20000], model, topk, self.stepsinfer) #added for the larger datasets later
                else:
                    mscore, mrank, recalls = getSimilarityScore(self.docs, model, topk, self.stepsinfer)

                #if istrain is False:
                #    print("Validation on %d images"%len(self.docs))
                print("Epoch %d, Mean Score: %f, Mean Rank: %s, Recalls: %s, Num documents: %d"%(self.epoch, mscore, str(mrank), str(recalls), len(self.docs)))

                #Write evaluation results to csv file
                with open(self.csvfile, 'a') as f:
                    writer = csv.writer(f, delimiter='\t')
                    writer.writerow([self.epoch, mscore, mrank, recalls['r@20'], recalls['r@50'], recalls['r@100'], len(self.docs)])

            self.epoch += 1

    return SimilarityCallback()

def getcallback_epochsaver(modeldir, epochnum): 
    class EpochSaver(CallbackAny2Vec):
        '''Callback to save model after each epoch.'''
        def __init__(self, modeldir, epochnum):
           self.modeldir = modeldir
           self.epoch = 1
           self.epochnum = epochnum

        def on_epoch_end(self, model):
            if self.epoch % self.epochnum == 0 and self.epoch != 0:
                num_docs = model.corpus_count
                dim = model.vector_size
                #fname = get_tmpfile(os.path.join(modeldir, 'g2vmodelc%dd%de%d'%(num_docs, dim, self.epoch)))
                #model.save(fname)
                with open(os.path.join(modeldir, 'g2vmodelc%dd%de%d'%(num_docs, dim, self.epoch)),'wb') as f:
                    dill.dump(model, f)
                print("Wrote model to file: ",os.path.join(modeldir, 'g2vmodel'))
            self.epoch += 1

    return EpochSaver(modeldir, epochnum)


def getcallback_logplot(outputpath, epochnum):  #TODO:implement seperate graphs and muliptle
    class PlotCallback(CallbackAny2Vec):
        def __init__(self):
            self.epoch = 1
            self.epochnum = epochnum
            self.imgpath = saveimgpath
            self.filepath =  os.path.join(outputpath, 'loss_log.jpg')

        def on_epoch_end(self, model):
            if self.epoch % self.epochnum == 0:
                print("Plotting Log")
                headers = ['Epoch', 'Train Loss', 'Train Mean Rank', 'Train Matched Docs', 'Val Loss', 'Val Mean Rank', 'Val Matched Docs']
                df = pd.read_csv(delimiter='\t', names=headers)

                def isfloat(x):
                    try:
                        x = float(x)
                        return True
                    except ValueError:
                        return False

                points = [(x,float(y)) for x,y in zip(df['Epoch'].to_list(), df['Train Loss'].to_list()) if isfloat(y)]
                x,y = zip(*points)
                plt.plot(x,y,label='Train Loss')

                points = [(x,float(y)) for x,y in zip(df['Epoch'].to_list(), df['Train Mean Rank'].to_list()) if isfloat(y)]
                x,y = zip(*points)
                plt.plot(x,y,label='Train Mean Rank')

                points = [(x,float(y)) for x,y in zip(df['Epoch'].to_list(), df['Train Matched Docs'].to_list()) if isfloat(y)]
                x,y = zip(*points)
                plt.plot(x,y,label='Train Matched Docs')


                points = [(x,float(y)) for x,y in zip(df['Epoch'].to_list(), df['Val Loss'].to_list()) if isfloat(y)]
                x,y = zip(*points)
                plt.plot(x,y,label='Val Loss')

                points = [(x,float(y)) for x,y in zip(df['Epoch'].to_list(), df['Val Mean Rank'].to_list()) if isfloat(y)]
                x,y = zip(*points)
                plt.plot(x,y,label='Val Mean Rank')

                points = [(x,float(y)) for x,y in zip(df['Epoch'].to_list(), df['Val Matched Docs'].to_list()) if isfloat(y)]
                x,y = zip(*points)
                plt.plot(x,y,label='Val Matched Docs')

                epochs = list(map(int, df['Epoch'].to_list()[1:]))
                plt.xticks(range(min(epochs), max(epochs)+1, 10))
                plt.legend(loc="upper left")
                plt.savefig(self.imgpath)
                plt.clf()

            self.epoch += 1

    return PlotCallback()


def getvaldocs(graphs, args):
    #Get an partially unseen validation set by modifying some node classes
    #Image name stay the same to allow for similarity evaluation
    #Important parameters:
    #   - _NUM_EXCHANGE_LABEL: Fraction of labels from the union of bbox & rels prediction labels,
    #                          which should be resampled randomly from all valid labels
    from validlabels import ind_to_classes, ind_to_predicates, VALID_BBOXLABELS, VALID_RELLABELS

    _NUM_EXCHANGE_LABEL = 0.2
    val_numimgs = int(len(graphs)*args.valsize)
    graphs = graphs[:val_numimgs]
    for filename, g in graphs:
        exchangenum = int(len(g['features'])* _NUM_EXCHANGE_LABEL)
        sample = dict(random.sample(g['features'].items(), exchangenum))
        for k,v in sample.items():
            if v in VALID_BBOXLABELS:
                sample[k] = random.choice(VALID_BBOXLABELS) 
            elif v in VALID_RELLABELS:
                sample[k] = random.choice(VALID_RELLABELS)
        g['features'].update(sample) 
    print("\nFeature extraction for validation dataset ({} graphs) started.\n".format(len(graphs)))
    document_collections = Parallel(n_jobs=args.workers)(delayed(feature_extractor)(gd[0], gd[1], args.wl_iterations) for gd in graphs) #tqdm(graphs))

    return document_collections

def getlogforlosses(outdir, istrain, topk): 
    filename = 'train_losses.csv' if istrain else 'val_losses.csv'
    with open(os.path.join(outdir, filename), 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        if istrain:
            headers = ['Epoch','Train Mean Score(@%d)'%topk, 'Train Mean Rank(@%d)'%topk, 'R@20' , 'R@50' , 'R@100', 'Num Docs'] 
        else:
            headers = ['Epoch', 'Val Mean Score(@%d)'%topk, 'Val Mean Rank(@%d)'%topk, 'R@20' , 'R@50' , 'R@100', 'Num Docs']
        writer.writerow(headers)
    return os.path.join(outdir, filename)

def saveconfig(output_dir, args, numtraindocs, doclenstats):
    #Writing config to file
    print(args)
    with open(os.path.join(output_dir, 'config.txt'), 'a') as f:
        f.write("Src Input Path: %s"%args.input_path + os.linesep)
        f.write("Vector dimensions: %s"%args.dimensions + os.linesep)
        f.write("Down-Sampling: %s"%args.down_sampling + os.linesep)
        f.write("Learning rate: %s"%args.learning_rate + os.linesep)
        f.write("Min count: %s"%args.min_count + os.linesep)
        f.write("Wl-iterations: %s"%args.wl_iterations + os.linesep)
        f.write("Steps Inference: %d"%args.steps_inference + os.linesep)
        f.write("Min Feature Dimension: %d"%args.min_featuredim + os.linesep)

        f.write("Epochs: %s"%args.epochs + os.linesep)
        f.write("Epochsave: %s"%args.epochsave + os.linesep)

        f.write("Number of graphs: %d"%numtraindocs + os.linesep)
        f.write("Statistics of feature number per graph:\n%s"%doclenstats + os.linesep)


    #Write entry to the csv overview file
    filepath = '/home/althausc/master_thesis_impl/graph2vec/models/run_configs.csv'
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            headers = ['Folder', 'Epochs', 'Trainsize', 'Valsize', 'Dim', 'Min-Count', 'WL-Iterations', 'LR', 'Down-Sampling', 'EvalTopk']
            writer.writerow(headers)

    foldername = os.path.basename(output_dir)       
    row = [foldername, args.epochs, numtraindocs, args.valsize, args.dimensions, args.min_count, args.wl_iterations,
                                args.learning_rate, args.down_sampling, args.evaltopk]
    with open(filepath, 'a') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(row)
    print("Sucessfully wrote hyper-parameter row to configs file.")

def evaluatemodel(docs, stepsinfer=100):
    print(str(getSimilarityScore(valdocs, model, topk, stepsinfer)))

def main(args):
    """
    Main function to read the graph list, extract features.
    Learn the embedding and save it.
    :param args: Object with the arguments.
    """
    #graphs = glob.glob(args.input_path + "*.json")
    
    data = None
    with open(args.input_path, "r") as f:
        data = json.load(f)  
    
    data2 = None
    if args.input_path_merge:
        with open(args.input_path_merge, "r") as f:
            data2 = json.load(f)  
        
    modeldir = os.path.join('/home/althausc/master_thesis_impl/graph2vec/models', datetime.datetime.now().strftime('%m-%d_%H-%M-%S')) 
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    else:
        raise ValueError("Output directory %s already exists."%modeldir)

    logdir = os.path.join(modeldir, '.logs')
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    else:
        raise ValueError("Output directory %s already exists."%logdir)

    graphs = list(data.items()) #[(filename, graph) ... , (filename, graph)]
                                #graph = {'edges': ... , 'features': ... , 'box_scores': ... , 'rel_scores': ...}
    if args.input_path_merge:
        graphs.extend(list(data2.items()))

    print("Number of training graphs: ",len(graphs))
    print("\nFeature extraction for train dataset ({} graphs) started.\n".format(len(graphs)))
    document_collections = Parallel(n_jobs=args.workers)(delayed(feature_extractor)(gd[0], gd[1], args.wl_iterations) for gd in graphs) #tqdm(graphs))

    if args.min_featuredim != -1:
        c_extended = 0
        for k,d in enumerate(document_collections):
            if len(d.words)<= args.min_featuredim:
                basefeatures = copy.deepcopy(d.words)
                add = False
                while len(d.words)<= args.min_featuredim:
                    d.words.extend(basefeatures)
                    add = True
                c_extended = c_extended + 1
        print("Extended {} features to minimal length of {}".format(c_extended, args.min_featuredim))

    #Stats to show words (features) per document (graph)
    doclengths = [len(d.words) for d in document_collections]
    doclenstats = getwhiskersvalues(doclengths)

    #Save configuration & stats
    saveconfig(modeldir, args, len(data), doclenstats)

    documents_validation = getvaldocs(graphs, args)
   
    c_eval_val = getcallback(documents_validation, epochnum=args.valeval, logdir= logdir, istrain=False, topk=args.evaltopk, stepsinfer=args.steps_inference)
    c_eval_train = getcallback(document_collections, epochnum=args.traineval, logdir= logdir, stepsinfer=args.steps_inference)
    c_epochsaver = getcallback_epochsaver(modeldir, args.epochsave)

    #Plot train & val curves at end of training
    #c_loss_plotter = getcallback_logplot(modeldir, args.epochs)

    #Training the Word2Vec model
    #Some parameters:
    #   sample (float, optional) – The threshold for configuring which higher-frequency words are randomly downsampled, useful range is (0, 1e-5).
    #   min_count (int, optional) – Ignores all words with total frequency lower than this.
    
    #Explanation of parameters:
    #   - document_collections: list of tagged documents (tagged document has fields tags & words)
    #   - min_count: exclude words with a frequency lower than this treshold from training
    #   - sample: threshold for configuring which higher-frequency words are randomly downsampled
    #   - epochs: Number of iterations (epochs) over the corpus


    #Notes:
    # -For a fully deterministically-reproducible run, you must also limit the model to a single worker thread (workers=1)
    # -Epoch values of 10-20 or more are most common in published results
    print("\nOptimization started.\n")
    model = Doc2Vec(document_collections,
                    vector_size=args.dimensions,
                    window=0,
                    min_count=args.min_count,
                    dm=0,
                    sample=args.down_sampling,
                    workers=args.workers,
                    epochs=args.epochs,
                    alpha=args.learning_rate,
                    compute_loss=True,
                    callbacks=[c_eval_train, c_eval_val, c_epochsaver])
    print(type(graphs))

    #save storage
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    filenames = [item[0] for item in graphs]
    
    graph_emb_file = os.path.join(modeldir, 'graph_embeddings.csv')
    save_embedding(graph_emb_file, model, filenames , args.dimensions)
    

    #Copy corresponding input training graph file to the model directory, for easier finding
    labelvectopk = os.path.join(os.path.dirname(args.input_path), 'labelvectors-topk.json')
    if os.path.isfile(labelvectopk):
        shutil.copyfile(labelvectopk, os.path.join(modeldir, 'labelvectors-topk.json'))
        print("Copied %s -> %s ."%(labelvectopk, os.path.join(modeldir, 'labelvectors-topk.json')))
    else:
        print("Couldn't copy labelvector file from %s, because not existing."%os.path.dirname(args.input_path))
   
if __name__ == "__main__":
    args = parameter_parser()
    main(args)
