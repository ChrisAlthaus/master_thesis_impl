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
import pickle
from collections import OrderedDict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('-descriptorFile','-descriptors',required=True,
                    help='Json file with keypoint descriptors in dicts with corresponding image id.')
parser.add_argument("-validateMethod", "-val", help="Helping wih choosing the right k for k-means.")
parser.add_argument("-modelState", "-model", help="Choosing model state of k-means clustering.")
parser.add_argument("-buildk", help="Building kmeans with specific k.", type=int)
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")

args = parser.parse_args()

if not os.path.isfile(args.descriptorFile):
    raise ValueError("No valid input file.")
if args.modelState is not None:
    if not os.path.isfile(args.modelState):
        raise ValueError("No valid input model file.")
if args.verbose:
    logging.basicConfig(level=logging.DEBUG)


def main():
    output_dir = os.path.join('/home/althausc/master_thesis_impl/posedescriptors/clustering/out', datetime.datetime.now().strftime('%m/%d_%H-%M-%S'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        raise ValueError("Output directory %s already exists."%output_dir)
   
    print("Reading from file: ",args.descriptorFile)
    with open (args.descriptorFile, "r") as f:
        json_data = json.load(f)

      
    if args.validateMethod is not None:
        descriptors = []

        for i, item in enumerate(json_data):
            descriptors.append(item['gpd'])
        descriptors = np.array(descriptors)

        if args.validateMethod == 'ELBOW':
            ks, sse = calculate_WSS(descriptors, 10, 170)
            
            print(ks)
            print(sse) # [4,20] = [2504852032.235484, 2409304817.4889417, 2316832857.527023, 2240042681.6462297, 2174669156.243677, 2105300280.3983455, 2053219336.5968208, 2014941360.954779, 1975832927.5999012, 1940372824.7540097, 1910655878.4689345, 1887375627.2133079, 1866513715.2268927, 1848455603.6091566, 1831448213.5342908, 1812663714.8990078, 1799162596.828914]
            
            #x = list(range(4,21))
            #y = [2504852032.235484, 2409304817.4889417, 2316832857.527023, 2240042681.6462297, 2174669156.243677, 2105300280.3983455, 2053219336.5968208, 2014941360.954779, 1975832927.5999012, 1940372824.7540097, 1910655878.4689345, 1887375627.2133079, 1866513715.2268927, 1848455603.6091566, 1831448213.5342908, 1812663714.8990078, 1799162596.828914]
            #range(10, 41)
            #[2053631804.047893, 2014946369.9704697, 1976370100.9062285, 1940954321.4685426, 1910455630.6114764, 1887624982.6240237, 1870007242.325744, 1848736676.252347, 1828779060.699307, 1812995629.8788753, 1797739167.9761512, 1782064767.217926, 1770218266.3849127, 1754376014.8209932, 1743390712.383553, 1731782005.7899482, 1716388995.4056718, 1709820655.3970628, 1699235431.66977, 1687935369.3011081, 1677652721.873943, 1671238990.609453, 1663928175.9675093, 1656810981.0248902, 1648482667.2718027, 1643671713.0389514, 1636561000.9212172, 1629881471.3745308, 1623287422.6246467, 1615720513.840895, 1612565871.966705]
            df = pd.DataFrame({'k':ks , 'sse':sse})
            ax = sns.relplot(x="k", y="sse", sort=False, kind="line", markers=True, data=df)
            ax.fig.savefig(os.path.join(output_dir,"eval_elbowmethod%d.png"%len(descriptors)))
            plt.clf()
        elif args.validateMethod == 'SILHOUETTE':
            """from sklearn.datasets import make_blobs
            X, y = make_blobs(n_samples=500,
                  n_features=2,
                  centers=4,
                  cluster_std=1,
                  center_box=(-10.0, 10.0),
                  shuffle=True,
                  random_state=1)  # For reproducibility
            ks, silouettes = calc_silouette_scores(X, 3, 6, plot_clustersilhouettes = True, output_dir= output_dir)"""
            ks, silouettes = calc_silouette_scores(descriptors, 10, 200, plot_clustersilhouettes = True, output_dir= output_dir)


            df = pd.DataFrame({'k':ks , 'silouette score':silouettes})
            ax = sns.relplot(x="k", y='silouette score', sort=False, kind="line", markers=True, data=df)
            ax.fig.savefig(os.path.join(output_dir,"eval_silouettes%d.png"%len(descriptors)))
            plt.clf()

        else:
            raise ValueError("Please specify a valid k-finding method.")
    else:
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

        print(v_codebook_assign)
        json_file = 'codebook_mapping'
        #with open(os.path.join(output_dir, json_file+'.json'), 'w') as f:
        #    print("Writing to file: ",os.path.join(output_dir,json_file+'.json'))
        #    json.dump(v_codebook_assign, f)
        #with open(os.path.join(output_dir, json_file+'.json'), 'w') as f:
        #    pickle.dump(v_codebook_assign, f)

        with open(os.path.join(output_dir, json_file+'.txt'), 'w') as f:
            f.write(str(v_codebook_assign)) 

def kmeans_and_visual_codebook(json_data, model=None, k=100):
    descriptors = []

    for item in json_data:
         print(item)
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

    #Create index mapping {gpd_cluster1: [{image_id: 1, score: 0.9, vis: [...]}, ..., {image_id: M, score: 0.78, vis: [...]}]}],
    #                        ...
    #                     {gpd_clusterK: [{image_id: 1, score: 0.9, vis: [...]}, ..., {image_id: L, score: 0.78, vis: [...]}]}] }
    v_codebook = {}
    map_codeword_ids = {}
    for i,centroid in enumerate(centroids):
        codeword = tuple(centroid)#(map(str, centroid))
        v_codebook[codeword] = []
        map_codeword_ids[i] = codeword
    print(map_codeword_ids)
 
    for i,cluster_id in enumerate(pred_clusters):
        #pred = list(filter(lambda item: item['image_id'] == img_id, json_data))[0]
        #if len(entry) > 1:
        #    raise IndexError("image_id should be unique.")
        entry = {'image_id': json_data[i]['image_id'], 'score': json_data[i]['score'], 'vis': json_data[i]['vis']}
        print("cluster_id: ",cluster_id)
        v_codebook[map_codeword_ids[cluster_id]].append(entry)

    print("Length of visual codebook: ",len(v_codebook))
    for i,c_list in enumerate(v_codebook):
        print("Cluster %d has %d gpd descriptors."%(i,len(c_list)))

    if model is None:
        return kmeans, v_codebook
    else:
        return None, v_codebook

# function returns WSS score for k values from 1 to kmax
def calculate_WSS(points, kmin, kmax):
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

def draw_acc_results():
    output_dir = os.path.join('/home/althausc/master_thesis_impl/posedescriptors/clustering/out', datetime.datetime.now().strftime('%m/%d_%H-%M-%S'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        raise ValueError("Output directory %s already exists."%output_dir)

    sse = [2053631804.047893, 2014946369.9704697, 1976370100.9062285, 1940954321.4685426, 1910455630.6114764,
        1887624982.6240237, 1870007242.325744, 1848736676.252347, 1828779060.699307, 1812995629.8788753, 1797739167.9761512,
        1782064767.217926, 1770218266.3849127, 1754376014.8209932, 1743390712.383553, 1731782005.7899482, 1716388995.4056718,
        1709820655.3970628, 1699235431.66977, 1687935369.3011081, 1677652721.873943, 1671238990.609453, 1663928175.9675093,
        1656810981.0248902, 1648482667.2718027, 1643671713.0389514, 1636561000.9212172, 1629881471.3745308, 1623287422.6246467,
        1615720513.840895, 1612565871.966705]
    sse.extend([1603722256.1261292, 1599764162.031898, 1593258003.4425516, 1586700139.6004117, 1582947281.2544641,
        1578755451.0539231, 1575487468.4650166, 1570923619.9702897, 1565599736.8021905, 1561114334.2671847, 1555746680.9366696,
        1550748872.7897816, 1545895943.4481752, 1544623353.4245994, 1540512065.7006834, 1533066729.6793535, 1530937271.0058255,
        1528188104.2423985, 1524597198.8899975, 1520017805.855355, 1517705623.3140378, 1514405016.5097027, 1508884485.7174535,
        1504838778.5315967, 1503705825.7196205, 1498172268.3246672, 1495414658.6343026, 1491767360.1945393,
        1491952658.418791, 1484120150.6439958])
    sse.extend([1483777424.5159192, 1484671937.8135004, 1479808823.3986971, 1476031099.5024536, 1469160539.240574,
                1467810991.2752087, 1466036947.6094024, 1464682059.1053398, 1458404131.9256685, 1458177838.4612713,
                1457263552.796704, 1453405196.6667717, 1449525947.222187, 1448356744.8128479, 1447429999.5468667,
                1443527948.52561, 1440033499.3951075, 1437686866.1304543, 1439855960.4113998, 1435528195.216776,
                1431974093.3771012, 1428763027.7969613, 1425468333.1430748, 1426248138.7511146, 1424985019.373451,
                1419944076.4704232, 1417117933.542482, 1414200159.633467, 1414214990.2169197, 1412181036.8249044,
                1408725646.221304, 1408191915.029111, 1406746150.681834, 1405395694.6419287, 1402657681.9242268,
                1399683178.5898404, 1397385380.589581, 1397667830.95049, 1393032593.8997984, 1393407083.7514002,
                1389171674.9817052, 1388245153.56092, 1386445153.4377117, 1384204944.115674, 1382868122.1062663,
                1380397912.1010554, 1379715169.0080156, 1378436607.3688276, 1374180579.1903276, 1375474282.5632489])
    sse.extend([1369383450.4107323, 1371288798.6049654, 1367815369.03859, 1364663645.4645832, 1366035665.7125661,
        1362480475.085779, 1364159769.1723487, 1362821549.306082, 1361184655.2618284, 1357057253.4299254, 1355748086.157733,
        1355473569.8905272, 1354019292.117037, 1352881762.6386933, 1349562848.9763255, 1348430988.5455852, 1348298287.563185,
        1343072164.0580542, 1344880050.5602455, 1343600051.8769202, 1340967976.5446727, 1337204068.602731, 1337097461.677344,
        1335643039.0756612, 1336930347.8174522, 1333350768.0020695, 1331167664.0333612, 1330944750.735233, 1328113014.9817457,
        1329292994.6845992, 1325669237.9838996, 1322368054.431572, 1324008588.8225486, 1321048739.4910343, 1319512122.0242996,
        1320340321.5772343, 1319250996.2377176, 1313860927.1961415, 1315702256.0598254, 1315624271.9877334, 1311311543.1945975,
        1310456875.3073435, 1311456673.0197616, 1308566110.2608488, 1302463591.3510082, 1304831784.5002322, 1305600316.8231888,
        1305041897.3674831, 1301507726.845836, 1302713185.9547124])
    
    ks = list(range(10,171))
    df = pd.DataFrame({'k':ks , 'sse':sse})
    ax = sns.relplot(x="k", y="sse", sort=False, kind="line", markers=True, data=df)
    ax.fig.savefig(os.path.join(output_dir,"eval_elbowmethod.png"))


if __name__=="__main__":
   main()
   #draw_acc_results()