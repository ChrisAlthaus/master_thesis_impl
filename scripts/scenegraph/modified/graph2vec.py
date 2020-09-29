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
import shutil
import dill
import random
import csv
import matplotlib.pyplot as plt

sgraphscript_dir = '/home/althausc/master_thesis_impl/scripts/scenegraph'
sys.path.insert(0,sgraphscript_dir) 

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
            features = [str(self.features[node])]+sorted([str(deg) for deg in degs])
            #print("f= ",features)
            features = "_".join(features)
            #print("f= ",features)
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            #print("h= ",hashing)
            new_features[node] = hashing
        self.extracted_features = self.extracted_features + list(new_features.values())
        #print("Extracted: ",self.extracted_features)
        
        return new_features

    def do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for _ in range(self.iterations):
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

    print("Wrote graph embedding to file: ",output_path)

def save_prediction_emb(output_path, emb, dimensions):  #not used
    assert len(emb) == dimensions
    column_names = ["x_"+str(dim) for dim in range(dimensions)]
    out = pd.DataFrame([emb], columns=column_names)
    out.to_csv(output_path, index=None)

    print("Wrote graph embedding to file: ",output_path)



def getSimilarityScore(valdocs, model, log, topk):
    #Querying with same images & specify similarities between the best-matching which should be the input image
    num_docs = topk
    dim = model.vector_size

    evalscore = 0
    numdocs_found = 0
    ranks = []

    for doc in valdocs:
        imgname = doc.tags[0]
        #model.random.seed(0)
        vector = model.infer_vector(doc.words)
        sims = model.docvecs.most_similar([vector], topn = num_docs)
        for r,item in enumerate(sims):
            if item[0] == imgname:
                evalscore = evalscore + item[1]
                numdocs_found = numdocs_found + 1
                ranks.append(r+1)
                break

    #print("Number of matched documents in top-%d = %d"%(num_docs, numdocs_found))
    #print("Mean rank number: ",sum(ranks)/len(ranks) if numdocs_found != 0 else 'Nan')
     
    mrank = sum(ranks)/len(ranks) if numdocs_found != 0 else 'Nan'
    return evalscore, mrank, numdocs_found
    
def getcallback(docs_collection, epochnum = 1, log=None, istrain=True, topk=100):
    #Monitoring the loss value of every epoch
    class SimilarityCallback(CallbackAny2Vec):
        def __init__(self):
            self.epoch = 1
            self.docs = docs_collection
            self.epochnum = epochnum
            self.log = log

        def on_epoch_end(self, model):
            if self.epoch % self.epochnum == 0:
                score, mrank, numfound = getSimilarityScore(self.docs, model, self.log, topk)
                if istrain is False:
                    print("Validation on %d images -> Epoch %d, Evaluation Score: %f, Mean Rank: %s, Num docs matched: %d/%d"\
                                                                        %(len(self.docs),self.epoch,score, str(mrank), numfound, len(self.docs)))
                else:
                    print("Epoch %d, Evaluation Score: %f, Mean Rank: %s, Num docs matched: %d/%d"%(self.epoch,score, str(mrank), numfound, len(self.docs)))

                with open(self.log, 'a') as f:
                    writer = csv.writer(f, delimiter='\t')
                    if istrain:
                        writer.writerow([self.epoch, score, mrank, numfound, 'notused', 'notused', 'notused'])
                    else:
                        writer.writerow([self.epoch, 'notused', 'notused', 'notused', score, mrank, numfound])
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


def getcallback_logplot(log, epochnum, saveimgpath):
    class PlotCallback(CallbackAny2Vec):
        def __init__(self):
            self.epoch = 1
            self.log = log
            self.epochnum = epochnum
            self.imgpath = saveimgpath

        def on_epoch_end(self, model):
            if self.epoch % self.epochnum == 0:
                print("Plotting Log")
                headers = ['Epoch','Train Loss', 'Train Mean Rank', 'Train Matched Docs', 'Val Loss', 'Val Mean Rank', 'Val Matched Docs']
                df = pd.read_csv(self.log, delimiter='\t', names=headers)

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
    from validlabels import ind_to_classes, ind_to_predicates, VALID_BBOXLABELS, VALID_RELLABELS

    _NUM_EXCHANGE_LABEL = 0.1
    graphs = graphs[:args.valsize]
    for filename, g in graphs:
        exchangenum = int(len(g['features'])* _NUM_EXCHANGE_LABEL)
        sample = dict(random.sample(g['features'].items(), exchangenum))
        for k,v in sample.items():
            if v in VALID_BBOXLABELS:
                sample[k] = random.choice(VALID_BBOXLABELS) 
            elif v in VALID_RELLABELS:
                sample[k] = random.choice(VALID_RELLABELS)
        g['features'].update(sample) 
    print("\nFeature extraction for validation dataset started.\n")
    document_collections = Parallel(n_jobs=args.workers)(delayed(feature_extractor)(gd[0], gd[1], args.wl_iterations) for gd in tqdm(graphs))
    print("\nOptimization started.\n")
    return document_collections

def getlogforlosses(outdir): 
    with open(os.path.join(outdir, 'trainval_losses.csv'), 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        headers = ['Epoch','Train Loss', 'Train Mean Rank', 'Train Matched Docs', 'Val Loss', 'Val Mean Rank', 'Val Matched Docs']
        writer.writerow(headers)
    return os.path.join(outdir, 'trainval_losses.csv')

def saveconfig(outdir, args, numtraindocs):
    with open(os.path.join(outdir, 'config.csv'), 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        headers = ['Epochs', 'Trainsize', 'Valsize', 'Dim', 'Min-Count', 'WL-Iterations', 'LR', 'Down-Sampling', 'EvalTopk']
        writer.writerow(headers)
        writer.writerow([args.epochs, numtraindocs, args.valsize, args.dimensions, args.min_count, args.wl_iterations,
                                args.learning_rate, args.down_sampling, args.evaltopk])

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
        
    modeldir = os.path.join('/home/althausc/master_thesis_impl/graph2vec/models', datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    else:
        raise ValueError("Output directory %s already exists."%modeldir)

    #Save configuration
    saveconfig(modeldir, args, len(data))
    
    graphs = list(data.items()) #[(filename, graph) ... , (filename, graph)]
                                #graph = {'edges': ... , 'features': ... , 'box_scores': ... , 'rel_scores': ...}
    print("Number of training graphs: ",len(graphs))
    print("\nFeature extraction for train dataset started.\n")
    document_collections = Parallel(n_jobs=args.workers)(delayed(feature_extractor)(gd[0], gd[1], args.wl_iterations) for gd in tqdm(graphs))
    print("\nOptimization started.\n")
    
    documents_validation = getvaldocs(graphs, args)
    logpath = getlogforlosses(modeldir)
    c_eval_val = getcallback(documents_validation, epochnum=args.valeval, log=logpath, istrain=False, topk=args.evaltopk)
    c_eval_train = getcallback(document_collections, log=logpath)
    c_epochsaver = getcallback_epochsaver(modeldir, args.epochsave)

    plotfile = os.path.join(modeldir, 'loss_log.jpg')
    c_loss_plotter = getcallback_logplot(logpath, args.plotepoch, plotfile)

    #Some parameters:
    #   sample (float, optional) – The threshold for configuring which higher-frequency words are randomly downsampled, useful range is (0, 1e-5).
    #   min_count (int, optional) – Ignores all words with total frequency lower than this.
    #Training the Word2Vec model
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
                    callbacks=[c_eval_train, c_eval_val, c_epochsaver, c_loss_plotter])
    print(type(graphs))

    #save storage
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    filenames = [item[0] for item in graphs]
    
    graph_emb_file = os.path.join(modeldir, 'graph_embeddings.csv')
    save_embedding(graph_emb_file, model, filenames , args.dimensions)
    


    #Copy corresponding input training graph file to the model directory, for easier finding
    labelvectopk = os.path.join(os.path.dirname(args.input_path), 'labelvectors_topk.json')
    if os.path.isfile(labelvectopk):
        shutil.copyfile(labelvectopk, os.path.join(modeldir, 'labelvectors_topk.json'))
        print("Copied %s -> %s ."%(labelvectopk, os.path.join(modeldir, 'labelvectors_topk.json')))
    else:
        print("Couldn't copy labelvector file from %s, because not existing."%os.path.dirname(args.input_path))
        
if __name__ == "__main__":
    args = parameter_parser()
    main(args)
