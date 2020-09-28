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
import os
import datetime
import shutil
import dill

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



def getSimilarityScore(traindocs, model):
    #Querying with same images & specify similarities between the best-matching which should be the input image
    num_docs = 100
    dim = model.vector_size

    evalscore = 0
    numdocs_found = 0
    ranks = []

    for doc in traindocs:
        imgname = doc.tags[0]
        #model.random.seed(0)
        vector = model.infer_vector(doc.words)
        sims = model.docvecs.most_similar([vector], topn = num_docs)
        for r,item in enumerate(sims):
            if item[0] == imgname:
                evalscore = evalscore + item[1]
                numdocs_found = numdocs_found + 1
                ranks.append(r)
                break
    print("Number of documents in train set: ",len(traindocs))
    print("Number of matched documents in top-%d = %d"%(num_docs, numdocs_found))
    print("Mean rank number: ",sum(ranks)/len(ranks))
    return evalscore
    
def getcallback(traindocs):
    #Monitoring the loss value of every epoch
    class SimilarityCallback(CallbackAny2Vec):
        def __init__(self):
            self.epoch = 1

        def on_epoch_end(self, model):
            score = getSimilarityScore(traindocs, model)
            print("Evaluation score in epoch %d = %f"%(self.epoch, score))
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
    
    graphs = list(data.items()) #[(filename, graph) ... , (filename, graph)]
                                #graph = {'edges': ... , 'features': ... , 'box_scores': ... , 'rel_scores': ...}
    print("\nFeature extraction started.\n")
    document_collections = Parallel(n_jobs=args.workers)(delayed(feature_extractor)(gd[0], gd[1], args.wl_iterations) for gd in tqdm(graphs))
    print("\nOptimization started.\n")
    
    c_evaluation = getcallback(document_collections)
    c_epochsaver = getcallback_epochsaver(modeldir, args.epochsave)

    
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
                    callbacks=[c_evaluation, c_epochsaver])
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
