import os
import json 
import argparse
import numpy as np
import math
import datetime
import time
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from collections import OrderedDict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Used to build a clustering based on the k-means algorithm from the input GPD descriptors.
#An evaluation option creates evaluation visualizations for choosing the right k.

parser = argparse.ArgumentParser()
parser.add_argument('-descriptorfile','-descriptors',required=True,
                    help='Json file with keypoint descriptors in dicts with corresponding image id.')
parser.add_argument("-validateMethod", "-val", help="Helping wih choosing the right k for k-means.")
parser.add_argument("-validateks", "-ks", nargs=2, type=int, help="Range for ks (kmin,kmax) used for evaluation method.")
parser.add_argument("-modelState", "-model", help="Choosing model state of k-means clustering.")
parser.add_argument("-buildk", help="Building kmeans with specific k.", type=int)
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")

args = parser.parse_args()

_VALIDATION_METHODS = ['ELBOW', 'SILHOUETTE', 'T-SNE', 'COS-TRESH']

if not os.path.isfile(args.descriptorfile):
    raise ValueError("No valid input file.")
if args.modelState is not None:
    if not os.path.isfile(args.modelState):
        raise ValueError("No valid input model file.")
if args.verbose:
    logging.basicConfig(level=logging.DEBUG)
if args.validateMethod not in _VALIDATION_METHODS:
    raise ValueError("No valid validation mode.")

def main():
    output_dir = '/home/althausc/master_thesis_impl/posedescriptors/clustering/%s'%('eval' if args.validateMethod is not None else 'out')
    output_dir = os.path.join(output_dir, datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        raise ValueError("Output directory %s already exists."%output_dir)
   
    print("Reading from file: ",args.descriptorfile)
    with open (args.descriptorfile, "r") as f:
        json_data = json.load(f)

    # ------------------------------- EVALUATION BASED ON NUMBER(S) OF CLUSTERS K ---------------------------------   
    if args.validateMethod is not None:
        descriptors = []

        for i, item in enumerate(json_data):
            descriptors.append(item['gpd'])
        descriptors = np.array(descriptors)

        if args.validateMethod == 'ELBOW':
            #Elbow method: > Within-Cluster-Sum of Squared Errors (WSS) for different values of k
            #              > idea: choose the k for which WSS first starts to diminish
            kmin, kmax = args.validateks
            ks, sse = calculate_WSS(descriptors, kmin, kmax)

            df = pd.DataFrame({'k':ks , 'sse':sse})
            ax = sns.relplot(x="k", y="sse", sort=False, kind="line", markers=True, data=df)
            ax.fig.savefig(os.path.join(output_dir,"eval_elbowmethod_c%dd_%d.png"%(len(descriptors), len(descriptors[0]))))
            plt.clf()

        elif args.validateMethod == 'SILHOUETTE':
            #Silhouette method: > Measures how similar a point is to its own cluster (cohesion) compared to other clusters (separation).
            #                   > Range is [-1,1] & higher is better
            """
            X = getdummydataset()
            ks, silouettes = calc_silouette_scores(X, 3, 6, plot_clustersilhouettes = True, output_dir= output_dir)"""
            kmin, kmax = args.validateks
            ks, silouettes = calc_silouette_scores(descriptors, kmin, kmax, plot_clustersilhouettes = True, output_dir= output_dir)

            df = pd.DataFrame({'k':ks , 'silouette score':silouettes})
            ax = sns.relplot(x="k", y='silouette score', sort=False, kind="line", markers=True, data=df)
            ax.fig.savefig(os.path.join(output_dir,"eval_silouettes_c%dd_%d.png"%(len(descriptors), len(descriptors[0]))))
            plt.clf()

        elif args.validateMethod == 'T-SNE':
            #Visualize the clustering of descriptors with t-sne algorithm
            k, _ = args.validateks
            X_embedded, labels = calc_tsne(descriptors, k)

            df = pd.DataFrame({'x':X_embedded[:,0] , 'y':X_embedded[:,1], 'labels': labels})
            fig, ax = plt.subplots(figsize=(16,10))
            g = ax.scatter(df['x'],df['y'], c=df['labels'], cmap=plt.get_cmap("jet",k), alpha=.7)
            fig.colorbar(g)
            fig.savefig(os.path.join(output_dir,"eval_tsne_c%dd_%d.png"%(len(descriptors), len(descriptors[0]))) )
            plt.clf()

        elif args.validateMethod == 'COS-TRESH':
            #Comparison based on cos-similarity, for checking gpds?!
            print("Computing cos-sim between all clusters ...")
            #Compute cos-sim of each gpd cluster with each other gpd cluster and visualize
            sim = cosine_similarity(descriptors, descriptors)
            print("Computing cos-sim between all clusters done.")

            x = []
            y = []
            for i,row in enumerate(sim):
                for j,s in enumerate(row):
                    if i == j:
                        x.append('Same')
                        y.append(s) 
                    else:
                        x.append('Different')
                        y.append(s)

            df = pd.DataFrame({'Cluster Similarity Mode':x, 'Cosine Similarities':y})
            ax = sns.boxplot(x='Cluster Similarity Mode', y='Cosine Similarities', data=df, flierprops = dict(markerfacecolor = '0.50', markersize = 0.5, marker='_'))
            #ax = sns.swarmplot(x='Cluster Similarity Mode', y='Cosine Similarities', data=df, size=1, color=".25")
            ax.figure.savefig(os.path.join(output_dir,"eval_simtresh_c%d.png"%sim[0].size))
            plt.clf()

            stats = df.groupby('Cluster Similarity Mode')['Cosine Similarities'].describe()
            with open(os.path.join(output_dir,"eval_statistics.txt"), "w") as f:
                f.write(str(stats))
            
        else:
            raise ValueError("Please specify a valid k-finding method.")
        
        saveconfig(output_dir,'eval',args.descriptorfile, len(descriptors))
        print("Output directory of evaluation: ", output_dir)

    # ------------------------------ BUILD OF CLUSTERING ----------------------------       
    else:
        #Either build the model with k-means and infering visual codebook or
        #use given model to infer visual codebook.
        #Visual Codebook: >Format: dict with elements which represent mappings between 
        #                              gpd cluster vector -> list of imagids+metadata
        #                 >Used later in the pose descriptor workflow for inserting to database

        v_codebook_assign = None
        if args.modelState is not None:
            #Infer mapping for a specific model state
            _, v_codebook_assign = kmeans_and_visual_codebook(json_data, model=args.modelState)
        elif args.buildk is not None:
            #Build kmeans with a specific k and infer mapping
            model, v_codebook_assign = kmeans_and_visual_codebook(json_data, k=args.buildk) 
            pickle.dump(model, open(os.path.join(output_dir,'modelk%d'%args.buildk + '.pkl'), "wb"))
        else:
            raise ValueError("Nothing to do. Please specify valid arguments.")

        json_file = 'codebook_mapping'
        with open(os.path.join(output_dir, json_file+'.txt'), 'w') as f:
            f.write(str(v_codebook_assign)) 

        saveconfig(output_dir,'build',args.descriptorfile, len(json_data), buildk=args.buildk, kmeansmod=args.modelfile)
        print("Output directory of clustering: ", output_dir)


def saveconfig(outputdir, mode, inputfile, numdescriptors, buildk=None, kmeansmod=None):
    if mode=='eval':
        with open(os.path.join(outputdir, 'evalconfig.txt'), 'a') as f:
            f.write('Input File: %s\n'%inputfile)
            f.write('Number Descriptors: %d\n'%numdescriptors)
    if mode=='build':
        with open(os.path.join(outputdir, 'buildconfig.txt'), 'a') as f:
            f.write('Input File: %s\n'%inputfile)
            f.write('Number Descriptors: %d\n'%numdescriptors)   
            f.write('Build k: %d\n'%buildk)
            f.write('K-Means Model: %s\n'%kmeansmod) 
               
def kmeans_and_visual_codebook(json_data, model=None, k=100):
    descriptors = []
    for item in json_data:
         descriptors.append(item['gpd'])
    descriptors = np.array(descriptors)
    
    if model is None:
        print("Clustering for k = %d ..."%k)
        start_time = time.time()
        kmeans = KMeans(n_clusters = k).fit(descriptors)
        print("Clustering for k = %d done. Took %s seconds."%(k,time.time() - start_time))  
    else:
        print("Opening k-means model %s ..."%args.modelState)
        kmeans = pickle.load(open(args.modelState, "rb"))
        print("Opening k-means model done.")
    
    centroids = kmeans.cluster_centers_
    pred_clusters = kmeans.predict(descriptors)

    #Visual Codebook output format: 
    #  { gpd_cluster1: [{image_id: 1, score: 0.9, vis: [...]}, ..., {image_id: M, score: 0.78, vis: [...]}]}],
    #                        ...
    #    gpd_clusterK: [{image_id: 1, score: 0.9, vis: [...]}, ..., {image_id: L, score: 0.78, vis: [...]}]}] }
    v_codebook = {}
    map_codeword_ids = {}
    for i,centroid in enumerate(centroids):
        codeword = tuple(centroid)#(map(str, centroid))
        v_codebook[codeword] = []
        map_codeword_ids[i] = codeword

    for i,cluster_id in enumerate(pred_clusters):
        entry = {'image_id': json_data[i]['image_id'], 'score': json_data[i]['score'], 'vis': json_data[i]['vis']}
        v_codebook[map_codeword_ids[cluster_id]].append(entry)

    #Print stats of visual codebook
    print("Length of visual codebook: ",len(v_codebook))
    for i,c_list in enumerate(v_codebook.values()):
        print("Cluster %d has %d gpd descriptors."%(i,len(c_list)))

    if model is None:
        return kmeans, v_codebook
    else:
        return None, v_codebook

def getdummydataset():
    #Dataset for debugging/sanity checking
    X, y = make_blobs(n_samples=500,
          n_features=2,
          centers=4,
          cluster_std=1,
          center_box=(-10.0, 10.0),
          shuffle=True,
          random_state=1)
    return X


def calculate_WSS(points, kmin, kmax):
    # Returns WSS score (Within-Cluster-Sum of Squared Errors) for k values from 1 to kmax
    sse = []
    for k in range(kmin, kmax+1):
        print("Clustering for k = %d ..."%k)
        start_time = time.time()
        kmeans = KMeans(n_clusters = k).fit(points)
        print("Clustering for k = %d done. Took %s seconds."%(k,time.time() - start_time))
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)

        curr_sse = 0

        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += np.linalg.norm(curr_center-points[i])**2
        sse.append(curr_sse)
    return range(kmin,kmax+1), sse

def calc_silouette_scores(points, kmin, kmax, plot_clustersilhouettes = False, output_dir=None):
    sil = []
    # minimum number of clusters should be 2
    for k in range(kmin, kmax+1):
        print("Clustering for k = %d ..."%k)
        start_time = time.time()
        kmeans = KMeans(n_clusters = k).fit(points)
        print("Clustering for k = %d done. Took %s seconds."%(k,time.time() - start_time))
        
        labels = kmeans.labels_
        sil.append(silhouette_score(points, labels, metric = 'euclidean'))

        if plot_clustersilhouettes:
            sil_values = []
            sample_silhouette_values = silhouette_samples(points, labels)
            for i in range(k):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
                sil_values.append(sum(ith_cluster_silhouette_values)/len(ith_cluster_silhouette_values))

            df = pd.DataFrame({'Cluster Label':list(range(k)) , 'Silhouette Score':sil_values})
            ax = sns.barplot(y='Cluster Label', x='Silhouette Score', data=df, orient = 'h')
            ax.figure.savefig(os.path.join(output_dir,"eval_cluster%dsilouettes.png"%k))
            plt.clf()
    
    return range(kmin,kmax+1), sil 

def calc_tsne(points,k):
    print("Dimension of a feature vector = %d "%len(points[0]))
    print("Calculate t-SNE ...")
    start_time = time.time()
    X_embedded = TSNE(n_components=2).fit_transform(points)
    print("Calculate t-SNE done. Took %s seconds."%(time.time() - start_time))
    print("Clustering for k = %d ..."%k)
    start_time = time.time()
    kmeans = KMeans(n_clusters = k).fit(points)
    print("Clustering for k = %d done. Took %s seconds."%(k,time.time() - start_time))

    labels = kmeans.labels_
    return X_embedded, labels

if __name__=="__main__":
   main()