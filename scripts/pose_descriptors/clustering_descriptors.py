import os
import json 
import argparse
import numpy as np
import math
import datetime
import time
import logging
from sklearn.cluster import KMeans

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('-descriptorFile','-descriptors',required=True,
                    help='Json file with keypoint descriptors in dicts with corresponding image id.')
parser.add_argument("-validateMethod", "-val", help="Helping wih choosing the right k for k-means.")
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")

args = parser.parse_args()

if not os.path.isfile(args.descriptorFile):
    raise ValueError("No valid input file.")
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

    descriptors = []

    for i, item in enumerate(json_data):
        descriptors.extend(item['gpd'])

    descriptors = np.array(descriptors)
    
    if args.validateMethod is not None:
        if args.val == 'ELBOW':
            ks, sse = calculate_WSS(descriptors, 121, 170)
            
            print(ks)
            print(sse) # [4,20] = [2504852032.235484, 2409304817.4889417, 2316832857.527023, 2240042681.6462297, 2174669156.243677, 2105300280.3983455, 2053219336.5968208, 2014941360.954779, 1975832927.5999012, 1940372824.7540097, 1910655878.4689345, 1887375627.2133079, 1866513715.2268927, 1848455603.6091566, 1831448213.5342908, 1812663714.8990078, 1799162596.828914]
            
            #x = list(range(4,21))
            #y = [2504852032.235484, 2409304817.4889417, 2316832857.527023, 2240042681.6462297, 2174669156.243677, 2105300280.3983455, 2053219336.5968208, 2014941360.954779, 1975832927.5999012, 1940372824.7540097, 1910655878.4689345, 1887375627.2133079, 1866513715.2268927, 1848455603.6091566, 1831448213.5342908, 1812663714.8990078, 1799162596.828914]
            #range(10, 41)
            #[2053631804.047893, 2014946369.9704697, 1976370100.9062285, 1940954321.4685426, 1910455630.6114764, 1887624982.6240237, 1870007242.325744, 1848736676.252347, 1828779060.699307, 1812995629.8788753, 1797739167.9761512, 1782064767.217926, 1770218266.3849127, 1754376014.8209932, 1743390712.383553, 1731782005.7899482, 1716388995.4056718, 1709820655.3970628, 1699235431.66977, 1687935369.3011081, 1677652721.873943, 1671238990.609453, 1663928175.9675093, 1656810981.0248902, 1648482667.2718027, 1643671713.0389514, 1636561000.9212172, 1629881471.3745308, 1623287422.6246467, 1615720513.840895, 1612565871.966705]
            df = pd.DataFrame({'k':ks , 'sse':sse})
            ax = sns.relplot(x="k", y="sse", sort=False, kind="line", markers=True, data=df)
            ax.fig.savefig(os.path.join(output_dir,"eval_elbowmethod.png"))
        elif args.val == 'SILHOUETTE':
            print("TODO")
        else:
            raise ValueError("Please specify a valid k-finding method.")
    else:
        v_codebook_assign = kmeans_and_visual_codebook(120, descriptors)

        json_file = 'codebook_mapping'
        with open(os.path.join(output_dir, json_file+'.json'), 'w') as f:
            print("Writing to file: ",os.path.join(output_dir,json_file+'.json'))
            json.dump(v_codebook_assign, f)

def kmeans_and_visual_codebook(k, points):
    print("Clustering for k = %d ..."%k)
    start_time = time.time()
    kmeans = KMeans(n_clusters = k).fit(points)
    print("Clustering for k = %d done. Took %s seconds."%(k,time.time() - start_time))
    centroids = kmeans.cluster_centers_
    pred_clusters = kmeans.predict(points)

    v_codebook = {}
    map_codeword_ids = {}
    for i,codeword in enumerate(centroids):
        v_codebook[tuple(codeword)] = []
        map_codeword_ids = {i:tuple(codeword)}

    
    for i in pred_clusters:
        entry = {points[i]['image_id'], points[i]['score'], points[i]['visibilities']}
        v_codebook[map_codeword_ids[i]].append(entry)

    print("Length of visual codebook: ",len(v_codebook))
    for i,c_list in enumerate(v_codebook):
        print("Cluster %d has %d gpd descriptors."%(i,len(c_list)))

    return v_codebook

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
    ks = list(range(10,121))
    df = pd.DataFrame({'k':ks , 'sse':sse})
    ax = sns.relplot(x="k", y="sse", sort=False, kind="line", markers=True, data=df)
    ax.fig.savefig(os.path.join(output_dir,"eval_elbowmethod.png"))


if __name__=="__main__":
   main()
   #draw_acc_results()