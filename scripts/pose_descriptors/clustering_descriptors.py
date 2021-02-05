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
from collections import OrderedDict, defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import sys
sys.path.append('/home/althausc/master_thesis_impl/scripts/pose_descriptors')
from utils import replace_unvalidentries, get_reference_feature, get_neutral_pose_feature, create_reference_feature

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

#Used to build a clustering based on the k-means algorithm from the input GPD descriptors.
#An evaluation option creates evaluation visualizations for choosing the right k.

parser = argparse.ArgumentParser()
parser.add_argument('-descriptorfile','-descriptors',required=True,
                    help='Json file with keypoint descriptors in dicts with corresponding image id.')
parser.add_argument("-validateMethod", "-val", help="Helping wih choosing the right k for k-means.")
parser.add_argument("-validateks", "-ks", nargs=2, type=int, help="Range for ks (kmin,kmax) used for evaluation method.")
parser.add_argument("-modelState", "-model", help="Choosing model state of k-means clustering.")
parser.add_argument("-buildk", help="Building kmeans with specific k.", type=int)
parser.add_argument("-gpdtype", type=str)
parser.add_argument("-imagedir", help="Base image directory for TSNE-visualization.", type=str)
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")

args = parser.parse_args()

_VALIDATION_METHODS = ['ELBOW', 'SILHOUETTE', 'T-SNE', 'COS-TRESH', 'K-MEANSIMAGES']

if not os.path.isfile(args.descriptorfile):
    raise ValueError("No valid input file.")
if args.modelState is not None:
    if not os.path.isfile(args.modelState):
        raise ValueError("No valid input model file.")
if args.verbose:
    logging.basicConfig(level=logging.DEBUG)
#if args.validateMethod not in _VALIDATION_METHODS:
#    raise ValueError("No valid validation mode.")

#python3.6 /home/althausc/master_thesis_impl/scripts/pose_descriptors/clustering_descriptors.py -descriptors /home/althausc/master_thesis_impl/posedescriptors/out/insert/12-18_16-44-58-JcJLdLLa_reduced-insert/geometric_pose_descriptor_c_53615_mJcJLdLLa_reduced_t0.05_f1_mkpt7n1.json -validateMethod K-MEANSIMAGES -validateks 30 40 -v -gpdtype JcJLdLLa_reduced
#python3.6 /home/althausc/master_thesis_impl/scripts/pose_descriptors/clustering_descriptors.py -descriptors /home/althausc/master_thesis_impl/posedescriptors/out/insert/12-18_16-44-58-JcJLdLLa_reduced-insert/geometric_pose_descriptor_c_53615_mJcJLdLLa_reduced_t0.05_f1_mkpt7n1.json -validateMethod SILHOUETTE -validateks 10 200 -v -gpdtype JcJLdLLa_reduced
#/home/althausc/master_thesis_impl/posedescriptors/out/insert/12-22_22-34-21-JLd_all-insert/geometric_pose_descriptor_c_68753_mJLd_all_direct_t0.05_f1_mkpt7n1.json
#/home/althausc/master_thesis_impl/posedescriptors/out/insert/12-23_11-28-42-jcrel_insert/geometric_pose_descriptor_c_68753_mJc_rel_t0.05_f1_mkpt7n1.json

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


    #create_reference_feature(json_data, mode='JLd_all_direct') #['JcJLdLLa_reduced', 'JLd_all_direct', 'JJo_reduced', 'Jc_rel']
    #exit(1)

    # ------------------------------- EVALUATION BASED ON NUMBER(S) OF CLUSTERS K ---------------------------------   
    if args.validateMethod is not None:
        descriptors = []
        imageids = []
        scores = []

        for i, item in enumerate(json_data):
            descriptors.append(item['gpd'])
            imageids.append(item['image_id'])
            scores.append(item['score'])
        print("Number of descriptors: ",len(descriptors))

        #Only consider images with 1 person detected for better/clearer cluster visualization 
        #For images with multiple persons take best scored person
        imgidstoinds = defaultdict(list)
        for i, x in enumerate(imageids):
            imgidstoinds[x].append(i)
        delinds = []
        for imagid,indices in imgidstoinds.items():
            if len(indices)>1:
                sindices = [(k,scores[k]) for k in indices]
                sindices = sorted(sindices, key=lambda x: x[1], reverse=True)
                delinds.extend([k for k,score in sindices[1:]])

        delinds.sort(reverse=True)
        #print(delinds[:100])
        for ind in delinds:
            del descriptors[ind]
            del imageids[ind]
        print("Number of descriptors reduced (1-person/image): ",len(descriptors))
        #exit(1)

        #Replace unvalid entries -1 with values of neutral pose gpd, because clustering sensitive to outlier -1
        #->other values in ranges [0,1] & [0,3.14]
        #referencefeature = get_neutral_pose_feature()
        referencefeature =  get_reference_feature(args.gpdtype)
        print("Replacing unvalid entries with reference featurevalues...")
        for i,descr in enumerate(descriptors):
            replace_unvalidentries(descr, referencefeature)
        descriptors = np.array(descriptors)#[:1000]
        imageids = np.array(imageids)#[:1000]

        if args.validateMethod == 'ELBOW':
            #Elbow method: > Within-Cluster-Sum of Squared Errors (WSS) for different values of k
            #              > idea: choose the k for which WSS first starts to diminish
            kmin, kmax = args.validateks
            ks, sse = calculate_WSS(descriptors, kmin, kmax, stepsize=10)

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
            ks, silouettes, silouettecoeffs = calc_silouette_scores(descriptors, kmin, kmax, stepsize=10, plot_clustersilhouettes=True, output_dir=output_dir)

            df = pd.DataFrame({'k':ks , 'Silhouette Score':silouettes})
            ax = sns.relplot(x="k", y='Silhouette Score', sort=False, kind="line", markers=True, data=df)
            ax.fig.savefig(os.path.join(output_dir,"eval_silouettes_c%dd_%d.svg"%(len(descriptors), len(descriptors[0]))))
            ax.fig.savefig(os.path.join(output_dir,"eval_silouettes_c%dd_%d.png"%(len(descriptors), len(descriptors[0]))))
            plt.clf()

            scatterpoints = np.array([[k,s]  for i,k in enumerate(ks) for s in silouettecoeffs[i]])

            df = pd.DataFrame({"Number of Clusters":scatterpoints[:,0] , 'Silhouette Coefficients':scatterpoints[:,1]})
            ax = sns.scatterplot(x="Number of Clusters", y='Silhouette Coefficients', data=df, alpha =0.4)
            plt.plot(ks, silouettes, marker='o', color='r')

            ratio = 1/1.3
            xleft, xright = ax.get_xlim()
            ybottom, ytop = ax.get_ylim()
            ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
            
            ax.get_figure().savefig(os.path.join(output_dir,"eval_silouettecoefficients_c%dd_%d.svg"%(len(descriptors), len(descriptors[0]))))
            ax.get_figure().savefig(os.path.join(output_dir,"eval_silouettecoefficients_c%dd_%d.png"%(len(descriptors), len(descriptors[0]))))
            plt.clf()



        elif args.validateMethod == 'K-MEANSIMAGES':
            #Visualize clusters by image grids
            k, _ = args.validateks
            #sampleindxs = np.random.choice(len(descriptors), size=10000, replace=False)
            #descriptors = descriptors[sampleindxs]
            #imageids = imageids[sampleindxs]

            labels, distancemat, scoefs = calc_kmeans(descriptors, k)
            #print(set(labels))
            #print(distancemat[0])

            validimgids = os.listdir('/home/althausc/master_thesis_impl/detectron2/out/art_predictions/train/12-14_18-27-33/.visimages')


            labeltoimgids = {}
            for i,cluster_l in enumerate(labels):
                if '{}_overlay.jpg'.format(imageids[i]) not in validimgids:
                    continue
                cdist = distancemat[i][cluster_l]
                if cluster_l not in labeltoimgids:
                    labeltoimgids[cluster_l] = [(imageids[i], cdist)]
                else:
                    labeltoimgids[cluster_l].append((imageids[i], cdist))
            #print(list(labeltoimgids.items())[:2])

            #imagefiles = imagefiles[np.random.choice(len(imagefiles), size=nrows*ncolumns, replace=False)]
            rankedchunks = False#True
            labeltoclusterdata = {}
            nrows = 4
            ncolumns = 5
            firstn_summarized = 2 #first n rows are together
            

            for cluster_l, items in labeltoimgids.items():
                items.sort(key=lambda x: x[1])
                items = np.array(items)
                cdata = {'ids':[], 'imagenames':[], 'cdistances':[]}

                if rankedchunks:
                    itemchunks = np.array_split(items, nrows - (firstn_summarized - 1))
                    ids = np.array_split(range(len(items)), nrows - (firstn_summarized - 1))
                    added_firstn = False
                    
                    for rids, imchunk in zip(ids, itemchunks):
                        cdata['ids'].extend(rids[:ncolumns])
                        cdata['imagenames'].extend(imchunk[:,0][:ncolumns])
                        cdata['cdistances'].extend(imchunk[:,1][:ncolumns])

                        if not added_firstn:
                            cdata['ids'].extend(rids[ncolumns:ncolumns*2])
                            cdata['imagenames'].extend(imchunk[:,0][ncolumns:ncolumns*2])
                            cdata['cdistances'].extend(imchunk[:,1][ncolumns:ncolumns*2])
                            added_firstn = True

                    labeltoclusterdata[cluster_l] = cdata
                else:
                    rids = np.random.choice(len(items), size=nrows*ncolumns, replace=False)
                    cdata['ids'].extend(list(range(nrows*ncolumns))) #rids[:ncolumns])
                    cdata['imagenames'].extend(items[:,0][:ncolumns*nrows])
                    cdata['cdistances'].extend(items[:,1][:ncolumns*nrows])
                    labeltoclusterdata[cluster_l] = cdata 
        
            c = 0
            drawkpts = True
            if drawkpts:
                imagedir = '/home/althausc/master_thesis_impl/detectron2/out/art_predictions/train/12-14_18-27-33/.visimages'
            else:
                imagedir = args.imagedir

            for cluster_l, clusterdata in labeltoclusterdata.items():
                #print(cluster_l, clusterdata)
                outputfile = os.path.join(output_dir, "imagegrid_cl{}_sc{:0.2f}.png".format(cluster_l,scoefs[cluster_l]))
                title = "Sampled Images from Cluster {} (Silhouette Score:{:0.2f})".format(cluster_l, scoefs[cluster_l])
                plotImageGrid(clusterdata, title, nrows, ncolumns, imagedir, outputfile, drawkpts=drawkpts)
                #c = c + 1
                #if c == 4:
                #    break

        elif args.validateMethod == 'T-SNE':
            #Visualize the clustering of descriptors with t-sne algorithm
            k, _ = args.validateks
            #Only consider a subset of the descriptors because of computational costs
            sampleindxs = np.random.choice(len(descriptors), size=10000, replace=False)
            descriptors = descriptors[sampleindxs]
            imageids = imageids[sampleindxs]

            X_embedded, labels = calc_tsne(descriptors, k)

            #Visualize random sampled T-SNE points on grid
            df = pd.DataFrame({'x':X_embedded[:,0] , 'y':X_embedded[:,1], 'labels': labels})
            fig, ax = plt.subplots(figsize=(16,10))
            g = ax.scatter(df['x'],df['y'], c=df['labels'], cmap=plt.get_cmap("jet",k), alpha=.7)
            fig.colorbar(g)
            fig.savefig(os.path.join(output_dir,"eval_tsne_c%dd_%d.png"%(len(descriptors), len(descriptors[0]))) )
            plt.clf()

            #Visualize random samples images on 2D plane according to T-SNE points
            #Normalize to range [0,1]
            tx, ty =  X_embedded[:,0],  X_embedded[:,1]
            tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
            ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

            imageid_tnse = np.array(list(zip(imageids, tx, ty)))
            imageid_tnse = imageid_tnse[np.random.choice(len(imageids), size=1000, replace=False)]

            width = 4000
            height = 3000
            max_dim = 100
            #print(imageid_tnse)

            full_image = Image.new('RGBA', (width, height))
            for imgid, x, y in imageid_tnse:
                tile = Image.open(os.path.join(args.imagedir, '{}.jpg'.format(imgid)))
                if tile is None:
                    print("No image at: {}".format(os.path.join(args.imagedir, img)))
                    continue
                rs = max(1, tile.width/max_dim, tile.height/max_dim)
                tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)

                full_image.paste(tile, (int((width-max_dim)*float(x)), int((height-max_dim)*float(y))), mask=tile.convert('RGBA'))

            full_image.save(os.path.join(output_dir, "tsne_images.png"))

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


def calculate_WSS(points, kmin, kmax, stepsize=1):
    # Returns WSS score (Within-Cluster-Sum of Squared Errors) for k values from 1 to kmax
    sse = []
    for k in range(kmin, kmax+1, stepsize):
        print("Clustering for k = %d ..."%k)
        start_time = time.time()
        kmeans = KMeans(n_clusters = k, init='k-means++', random_state=1234).fit(points)
        print("Clustering for k = %d done. Took %s seconds."%(k,time.time() - start_time))
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)

        curr_sse = 0

        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += np.linalg.norm(curr_center-points[i])**2
        print("SSE score: ", curr_sse)
        sse.append(curr_sse)
    return range(kmin,kmax+1,stepsize), sse

def calc_silouette_scores(points, kmin, kmax, stepsize=1, plot_clustersilhouettes = False, output_dir=None):
    sil = [] #total score
    sil_values_per_cluster = [] # all cluster scores
    # minimum number of clusters should be 2
    for k in range(kmin, kmax+1, stepsize):
        print("Clustering for k = %d ..."%k)
        start_time = time.time()
        kmeans = KMeans(n_clusters = k, init='k-means++', random_state=1234).fit(points)
        print("Clustering for k = %d done. Took %s seconds."%(k,time.time() - start_time))
        
        labels = kmeans.labels_
        silscore = silhouette_score(points, labels, metric = 'euclidean')
        print("Silhouette Score: ", silscore)
        sil.append(silscore)

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
            sil_values_per_cluster.append(sil_values)
    
    return range(kmin,kmax+1,stepsize), sil, sil_values_per_cluster

def calc_tsne(points,k):
    print("Number of input features = %d"%len(points))
    print("Dimension of a feature vector = %d "%len(points[0]))
    print("Calculate t-SNE ...")
    start_time = time.time()
    #X_embedded = TSNE(n_components=2, verbose=1).fit_transform(points)
    X_embedded = None
    print("Calculate t-SNE done. Took %s seconds."%(time.time() - start_time))
    print("Clustering for k = %d ..."%k)
    start_time = time.time()
    kmeans = KMeans(n_clusters = k, init='k-means++', random_state=1234).fit(points)
    print("Clustering for k = %d done. Took %s seconds."%(k,time.time() - start_time))

    labels = kmeans.labels_
    return X_embedded, labels

def calc_kmeans(points,k):
    print("Clustering for k = %d ..."%k)
    start_time = time.time()
    kmeans = KMeans(n_clusters = k, init='k-means++', random_state=1234).fit(points)
    print("Clustering for k = %d done. Took %s seconds."%(k,time.time() - start_time))
    labels = kmeans.labels_
    print("Number of resulting clusters = %d"%len(set(labels)))
    distancemat = kmeans.transform(points)

    sil_values = {}
    sample_silhouette_values = silhouette_samples(points, labels)
    for i in range(k):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        sil_values[i] = sum(ith_cluster_silhouette_values)/len(ith_cluster_silhouette_values)
    return labels, distancemat, sil_values
    


def plotImageGrid(clusterdata, title, nrows, ncolumns, imgdir, savepath, drawkpts=False):
    fig = plt.figure(figsize=(4.0, 4.0))
    #fig.suptitle(title, y=0.9, fontsize=5)
    fig.tight_layout()
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(nrows, ncolumns),  # creates 2x2 grid of axes
                    axes_pad=0.15,  # pad between axes in inch.
                    share_all=True
                    )   

    basewidth = 400
    images = []
    for rid, filename, cdist in zip(clusterdata['ids'], clusterdata['imagenames'], clusterdata['cdistances']):
        if drawkpts:
            filepath = os.path.join(imgdir, '{}_overlay.jpg'.format(filename))
        else:
            filepath = os.path.join(imgdir, '{}.jpg'.format(filename))
        img = Image.open(filepath)
        #print(rid, img.size)
        #img = resizeimage(img)
        #print(img.size)
        images.append(np.array(img))
    
    c_added = 0
    """
    for ax, im, rid, cdist in zip(grid, images, clusterdata['ids'], clusterdata['cdistances']):
        # Iterating over the grid returns the Axes.
        ax.axis('off')
        imtitle = 'r={} d={:0.2f}'.format(rid, float(cdist))
        #ax.set_title(imtitle, fontdict=None, loc='center', color = "k", y=-0.01)
        ax.imshow(im)
        
        #ax.set_xlabel(imtitle)
        ax.text(0.5,-0.1, imtitle, size=2, ha="center", transform=ax.transAxes)
        c_added = c_added + 1"""

    _, axs = plt.subplots(nrows, ncolumns, figsize=(12, 12))
    axs = axs.flatten()
    for img, ax, rid, cdist in zip(images, axs, clusterdata['ids'], clusterdata['cdistances']): 
        #ax.set_anchor('NW')
        ax.imshow(img)
        #ax.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        imtitle = 'r={} d={:0.2f}'.format(rid, float(cdist))
        ax.set_xlabel(imtitle, labelpad = 4, fontsize=12) #fontweight='bold'
        c_added = c_added + 1
    plt.subplots_adjust(wspace=0.05, hspace=0.12)
    #if c_added<nrows*ncolumns:
    #    for i in range(c_added,nrows*ncolumns):
    #        grid[i].axis('off')
    #        ax.text(0.5,-0.1, "Image", size=2, ha="center", transform=ax.transAxes)

    plt.savefig(savepath, dpi=400, bbox_inches='tight', pad_inches=0.01)
    plt.clf()

def resizeimage(image):
    MAX_SIZE = 400
    original_size = max(image.size[0], image.size[1])
    if original_size >= MAX_SIZE:
        if (image.size[0] > image.size[1]):
            resized_width = MAX_SIZE
            resized_height = int(round((MAX_SIZE/float(image.size[0]))*image.size[1])) 
        else:
            resized_height = MAX_SIZE
            resized_width = int(round((MAX_SIZE/float(image.size[1]))*image.size[0]))
        image = image.resize((resized_width, resized_height), Image.ANTIALIAS)
    return image

if __name__=="__main__":
   main()