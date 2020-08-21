import os
import json 
import argparse
import numpy as np
import math
import datetime
import time
import logging
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser()
parser.add_argument('-descriptorFile','-descriptors',required=True,
                    help='Json file with keypoint descriptors in dicts with corresponding image id.')
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

    ks, sse = calculate_WSS(descriptors, 4, 20)
    
    print(ks)
    print(sse)

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


if __name__=="__main__":
   main()