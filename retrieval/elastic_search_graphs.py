import datetime
from elasticsearch import Elasticsearch, RequestsHttpConnection
from elasticsearch.helpers import scan
import elasticsearch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import itertools
import os
import csv

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import time
import argparse
import json
import logging
import ast

parser = argparse.ArgumentParser()
parser.add_argument('-file',required=True,
                    help='Json file with clustering centroids as keys and a list of image metadata as values.\
                    Or for search a dict of image descriptors')
parser.add_argument('-search_data','-search', action="store_true",
                    help='Whether to search the index for the input graph embedding data.')
parser.add_argument('-insert_data','-insert', action="store_true",
                    help='Whether to build an index from the cluster mapped data.')
"""
parser.add_argument('-eval_tresh','-evaltresh', action="store_true",
                    help='Computing cos-sim between gpd clusters to approximate a cutoff threshold.')
parser.add_argument('-method_ins', help='Select method for insert data.')"""
parser.add_argument('-method_search', default='standard', help='Select method for searching data.')
parser.add_argument('-tresh', type=float, help='Similarity treshold for .')
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")

args = parser.parse_args()

logger = logging.getLogger('elasticsearch-db')
if args.verbose:
    logger.setLevel(logging.DEBUG)
#logging.basicConfig(level=logging.DEBUG, format='%(message)s')

_INDEX = 'imgid_graphemb'
#Min score used for cossim
if args.tresh is None:
    _SIMILARITY_TRESH = 0.95
else:
    _SIMILARITY_TRESH = args.tresh

_METHODS_SEARCH = ['standard', 'custom']
_ELEMNUM_SEARCH = 100
_NUMRES = 10


def main():
    print("Reading from file: ",args.file)
    

    with open(args.file, newline='') as csvfile:
        data = list(csv.reader(csvfile, delimiter=','))

    print(len(data))
    print(data[0])
    print(data[1])

    if args.insert:
        #format of graph embeddings: type,x_0,x_1,x_2,x_3,..,x_n
        #skip header
        for h in data[0]:
            assert isinstance(h,str)
        del data[0]
        data = [[float(x) if i>0 else x for i,x in enumerate(row)] for row in data]

    output_dir = None
    if args.search_data:
        output_dir = os.path.join('/home/althausc/master_thesis_impl/retrieval/out/scenegraphs', datetime.datetime.now().strftime('%m/%d_%H-%M-%S'))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            raise ValueError("Output directory %s already exists."%output_dir)

    image_dir = '/home/althausc/nfs/data/vg_styletransfer/VG_100K'

    es = Elasticsearch("http://localhost:9200", #30000
                       ca_certs=False,
                       verify_certs=False)

    #print(get_alldocs(es))
    #exit(1)
    
    if args.insert_data:
        createIndex(es, len(data[0])-1)
        insertdata(args, data, es)


    elif args.search_data:
        es.indices.refresh(index=_INDEX)
        #data format: {'1': [featurevector], ... , 'n': [featurevector]}
        imgids_final = []
        results = []
        print("Searching image descriptors from %s ..."%args.file)

        image_ids, scores = query(es, data[0], _ELEMNUM_SEARCH, args.method_search)
        results.append(list(zip(image_ids,scores)))
        print("Searching image descriptors done.")
        print("QUERY RESULTS: ",results)

        imgids_final = results

        if isinstance(imgids_final[0], tuple):
            [image_ids, rel_scores] = zip(*imgids_final)
            saveResults(list(image_ids), list(rel_scores), output_dir, image_dir)
        elif isinstance(imgids_final[0], int):
            saveResults(imgids_final, None, output_dir, image_dir)
        
            
def insertdata(args, data ,es):           
    id = 0
    print("Inserting image descriptors from %s ..."%args.file)
    for item in data:
        img_id = item[0]
        graph_emb = item[1:]
        insertdoc(es, graph_emb, {'image_id': img_id, 'score': 1.0}, id, 'graphemb')
        id = id + 1
        if id%100 == 0 and id != 0:
            logger.debug("{} image descriptors were inserted so far.".format(id))
            print("{} image descriptors were inserted so far.".format(id))
    print("Inserted %d image descriptors."%(id))
    print("Inserting image descriptors done.")


def createIndex(es, dim):
    mapping = {
        "mappings": {
            "properties": {
                "imageid": {
                    "type": "text"
                },
                "score": {
                    "type": "float"
                },
                "graphemb": {
                    "type": "dense_vector",
                    "dims": dim
                }
            }               
        }
    }

    if es.indices.exists(index=_INDEX):
        es.indices.delete(index=_INDEX, ignore=[400, 404])

    response = es.indices.create(
        index=_INDEX,
        body=mapping,
        #ignore=400 # ignore 400 already exists code
    )
    if response['acknowledged'] != True:
        raise ValueError('Index was not created')

def insertdoc(es, featurevec, metadata, id, featurelabel):
    #print(metadata)
    #print(featurevec)
    doc = {
    'imageid': metadata['image_id'],
    'score': metadata['score'],
    featurelabel: list(featurevec)
    }
    #print(doc)
    res = ''
    try :
        res = es.index(index=_INDEX, id=id, body=doc) 
    except elasticsearch.ElasticsearchException as es1:  
        print(es1,es1.info)

    #print("inserting ",doc)
    if res['result'] != 'created':
        raise ValueError("Document not created.")

def query(es, featurevector, size, method):
    if method == _METHODS_SEARCH[0]:
        request = { "size": size,
                               #"min_score": _SIMILARITY_TRESH + 1, 
                               "query": {
                                "script_score": {
                                "query": {
                                    "match_all": {}
                                },
                                "script": {
                                    "source": "cosineSimilarity(params.queryVector, doc['graphemb'])+1.0",
                                    "params": {
                                    "queryVector": list(featurevector)  
                                    }
                                }
                                }
                            }
                        }
    else: # method == _METHODS[1], L2-Distance
        request = { "size": size,
                      "query": {
                       "script_score": {
                       "query": {
                           "match_all": {}
                       },
                       "script": {
                           "source": "l2norm(params.queryVector, doc['graphemb'])",
                           "params": {
                           "queryVector": list(featurevector)  
                           }
                        }
                       }
                      }
                  }
    try :
        res= es.search(index=_INDEX, 
                        body=request)
    except elasticsearch.ElasticsearchException as es1:  
        print("Error when querying for feature vector ",featurevector)
        print(es1,es1.info)

    print("Returned: ", request)
    exit(1)

    logger.debug("Query returned {} results stated.".format(res['hits']['total']['value'])) 
    logger.debug("Query returned {} results actual.".format(len(res['hits']['hits']))) 
    
    print("Query returned {} results stated.".format(res['hits']['total']['value']))
    print("Query returned {} results actual.".format(len(res['hits']['hits'])))
    docs = res['hits']['hits']
    imageids = [item['_source']['imageid'] for item in docs]
    scores = [item['_score'] for item in docs] 
     
    return imageids, scores



def get_alldocs(es):
    es.indices.refresh(index=_INDEX)
    response = scan(es, index=_INDEX, query={"query": { "match_all" : {}}})
    return list(response)

def saveResults(image_ids, rel_scores, output_dir, image_dir):
    imagemetadata = {'imagedir': image_dir}
    if rel_scores is None:
        for rank, imageid in enumerate(image_ids): 
            imgpath = "%s.jpg"%(str(imgid))
            imgpath = os.path.join(image_dir, imgpath)
            imagemetadata[rank] = {'filepath': imgpath}
    else:
        for rank, (imageid, relscore) in enumerate(zip(image_ids, rel_scores)):
            imgpath = "%s.jpg"%(str(imgid))
            imgpath = os.path.join(image_dir, imgpath)
            imagemetadata[rank] = {'filepath': imgpath, 'relscore': relscore}
    
    json_file = 'result-ranking'
    with open(os.path.join(output_dir, json_file+'.json'), 'w') as f:
        print("Writing to file: ",os.path.join(output_dir,json_file+'.json'))
        json.dump(imagemetadata, f, indent=4, separators=(',', ': '))
    print("Wrote %d ranked items to file."%len(image_ids))


if __name__=="__main__":
    main()

