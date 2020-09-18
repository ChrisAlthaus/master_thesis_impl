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
    
    graphs = list(data.items()) #[(filename, graph) ... , (filename, graph)]
    print("\nFeature extraction started.\n")
    document_collections = Parallel(n_jobs=args.workers)(delayed(feature_extractor)(gd[0], gd[1], args.wl_iterations) for gd in tqdm(graphs))
    print("\nOptimization started.\n")
    
    model = Doc2Vec(document_collections,
                    vector_size=args.dimensions,
                    window=0,
                    min_count=args.min_count,
                    dm=0,
                    sample=args.down_sampling,
                    workers=args.workers,
                    epochs=args.epochs,
                    alpha=args.learning_rate)
    print(type(graphs))
    filenames = [item[0] for item in graphs]
    save_embedding(args.output_path, model, filenames , args.dimensions)

if __name__ == "__main__":
    args = parameter_parser()
    main(args)
