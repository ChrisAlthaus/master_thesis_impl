import numpy as np
import random
from scipy import spatial
import argparse
import os
import datetime
import json
import copy
from numpy import dot
from numpy.linalg import norm

#Script used to generate multiple dummy gpd descriptor data for comparing the cossim scores.
#Result file can be used to query the db. Notice that only the first image descriptor of the input file is used as the base featurevector.

parser = argparse.ArgumentParser()
parser.add_argument('-file',type=str, help='Json file with image descriptors')
args = parser.parse_args()

with open (args.file, "r") as f:
    data = f.read()
data = eval(data)  
print(data)

dbdoc = data[0]
dbvec = dbdoc['gpd']
dbmask = dbdoc['mask']
querydocs = []

_NUM_QUERIES = 4
for i in range(_NUM_QUERIES):
    qvec = np.random.uniform(low=0.0, high=1.0, size=(len(dbvec),)).tolist()
    qmask = ''.join(random.choice('01') for _ in range(len(dbmask)))

    print(len(qvec), len(dbvec))
    v1 = copy.copy(dbvec)
    v2 = copy.copy(qvec) 
    for k in range(len(qmask)):
        if qmask[k] == '0' or dbmask[k] == '0':
            v2[k] = v1[k]

    result = 1 + dot(v1, v2)/(norm(v1)*norm(v2))

    #result = 1 + spatial.distance.cosine(v1, v2)
    querydocs.append({'gpd': qvec, 'mask': qmask, 'cossim': result, 'querynew': v2})

output_dir = os.path.join('/home/althausc/master_thesis_impl/posedescriptors/out/query', datetime.datetime.now().strftime('testsimilarity%m-%d_%H-%M-%S'))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)   

with open(os.path.join(output_dir, 'geometric_pose_descriptor.json'), 'w') as f:
        print("Writing to folder: ",output_dir)
        json.dump(querydocs, f, indent=4)