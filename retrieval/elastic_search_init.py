import datetime
from elasticsearch import Elasticsearch, RequestsHttpConnection
from elasticsearch.helpers import scan
import elasticsearch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import itertools
import os

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
parser.add_argument('-insert_data','-insert', action="store_true",
                    help='Whether to build an index from the cluster mapped data.')
parser.add_argument('-search_data','-search', action="store_true",
                    help='Whether to search the index for the input cluster data.')
parser.add_argument('-eval_tresh','-evaltresh', action="store_true",
                    help='Computing cos-sim between gpd clusters to approximate a cutoff threshold.')
parser.add_argument('-method_ins', help='Select method for insert data.')
parser.add_argument('-method_search', help='Select method for searching data.')
parser.add_argument('-tresh', type=float, help='Similarity treshold for cossim result ranking.')
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")

args = parser.parse_args()

logger = logging.getLogger('elasticsearch-db')
if args.verbose:
    logger.setLevel(logging.DEBUG)
#logging.basicConfig(level=logging.DEBUG, format='%(message)s')

_INDEX = 'imgid_gpdcluster'
#Min score used for cossim
if args.tresh is None:
    _SIMILARITY_TRESH = 0.95
else:
    _SIMILARITY_TRESH = args.tresh 
#Method 1 uses Cosinus-Similarity for comparing features with features in db produced by visual codebook preprocessing, 
#       just a shortlist of N best-matching documents is queried for better performance
#Method 2 uses a Distance-Measure to compute the sum between input features and raw features stored in the db (each similarity counts)
#       the image with smallest distance is selected
_METHODS_INS = ['CLUSTER', 'RAW']
_METHODS_SEARCH = ['COSSIM', 'DISTSUM']
_ELEMNUM_COS = 20
_NUMRES_DIST = 10

if args.method_search is not None:
    if args.method_search not in _METHODS_SEARCH:
        raise ValueError("Please specify a valid search method.") 
if args.method_ins is not None:
    if args.method_ins not in _METHODS_INS:
        raise ValueError("Please specify a valid insert method.")


def main():
    print("Reading from file: ",args.file)
    with open (args.file, "r") as f:
        data = f.read()
    data = eval(data)
    #print(data[:2])

    if args.insert_data: # or args.eval_tresh:
        """length = sum([len(buckets) for buckets in data.values()])
        print("Items in input data: ",length)

        for i in range(len(data)-4):
            del data[list(data.keys())[0]]
        
        for gpd, items in data.items():
            for i in range(len(items)-4):
                del items[0]
        print(len(data))

        length = sum([len(buckets) for buckets in data.values()])
        print("Items in input data reduced: ",length)"""
        #data = data[:1000]
        
    elif args.search_data:
         print("Items in input data: ",len(data))

    output_dir = None
    if args.search_data or args.eval_tresh:
        output_dir = os.path.join('/home/althausc/master_thesis_impl/retrieval/out', datetime.datetime.now().strftime('%m/%d_%H-%M-%S'))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            raise ValueError("Output directory %s already exists."%output_dir)

    image_dir = '/home/althausc/nfs/data/coco_17_medium/train2017_styletransfer'

    es = Elasticsearch("http://localhost:9200", #30000
                       ca_certs=False,
                       verify_certs=False)
    
    if args.insert_data:
        
        if args.method_ins == _METHODS_INS[0]:
            createIndex(es, len(list(data.keys())[0]), _METHODS_INS[0])
            insertdata_cluster(args, data, es)
        elif args.method_ins == _METHODS_INS[1]:
            createIndex(es, len(data[0]['gpd']), _METHODS_INS[1])
            insertdata_raw(args, data, es)


    elif args.search_data:
        es.indices.refresh(index=_INDEX)
        #data format: {'1': [featurevector], ... , 'n': [featurevector]}
        imgids_final = []
        if args.method_search == _METHODS_SEARCH[0]:
            results = []
            print("Searching image descriptors from %s ..."%args.file)
            for img_descriptor in data:
                print(img_descriptor)
                image_ids, scores = query(es, img_descriptor['gpd'], _ELEMNUM_COS, args.method_search)
                results.append(list(zip(image_ids,scores)))
            print("Searching image descriptors done.")
            print(results)
        
            #Flattening list of indiv feature vector results
            results = [item for sublist in results for item in sublist]
            imgids_final = bestmatching_cluster(results)
        
        elif args.method_search == _METHODS_SEARCH[1]:
            results = []
            numElemAll = es.cat.count(_INDEX, params={"format": "json"})[0]['count']
            print("Total documents in the index: %d."%int(numElemAll))
            print("Searching image descriptors from %s ..."%args.file)
            for img_descriptor in data:
                image_ids, scores = query(es, img_descriptor['gpd'], int(numElemAll), args.method_search)
                results.append(list(zip(image_ids,scores)))
            print("Searching image descriptors done.")
            
            #Flattening list of indiv feature vector results
            results = [item for sublist in results for item in sublist]
            imgids_final = bestmatching_sumdist(results, _NUMRES_DIST)
        print("Best matched images: ", imgids_final)

        if isinstance(imgids_final[0], tuple):
            [image_ids, rel_scores] = zip(*imgids_final)
            saveResults(list(image_ids), list(rel_scores), output_dir, image_dir)
        elif isinstance(imgids_final[0], int):
            saveResults(imgids_final, None, output_dir, image_dir)


    elif args.eval_tresh and args.method_search == _METHODS_SEARCH[0]:
        #Compute cos-sim of each gpd cluster with each other gpd cluster and visualize
        gpdclusters = []
        for gpdcluster, _ in data.items():
            gpdclusters.append(gpdcluster)
        gpdclusters = np.array(gpdclusters)
        sim = cosine_similarity(gpdclusters, gpdclusters)

        x = []
        y = []
        for i,row in enumerate(sim):
            for s in row:
                x.append(i)
                y.append(s)

        df = pd.DataFrame({'Source GPD cluster':x , 'Cosine Similarities':y})
        #ax = sns.regplot(x='Source GPD cluster', y='Cosine Similarities', fit_reg=False, data=df)
        
        ax = sns.boxplot(x='Source GPD cluster', y='Cosine Similarities', data=df)
        ax = sns.swarmplot(x='Source GPD cluster', y='Cosine Similarities', data=df, size=2, color=".25")
        ax.figure.savefig(os.path.join(output_dir,"eval_simtresh_c%d.png"%sim[0].size))
        plt.clf()
        
            
def insertdata_raw(args, data ,es):           
    id = 0
    print("Inserting image descriptors from %s ..."%args.file)
    for item in data:
        insertdoc(es, item['gpd'], {'image_id': item['image_id'], 'score': item['score']}, id, 'gpd')
        id = id + 1
        if id%100 == 0 and id != 0:
            logger.debug("{} image descriptors were inserted so far.".format(id))
            print("{} image descriptors were inserted so far.".format(id))
    print("Inserted %d image descriptors."%(id))
    print("Inserting image descriptors done.")
     
def insertdata_cluster(args, data, es):
   #data format: {gpdcluster1: [{img_id, score, vis}, ... ,{img_id, score, vis}], ... , gpdclusterK: [{img_id, score, vis}, ... ,{img_id, score, vis}]}
    id = 0
    print("Inserting image descriptors from %s ..."%args.file)
    for gpdcluster, imgs_metadata in data.items():
        for metadata in imgs_metadata:
            insertdoc(es, gpdcluster, metadata, id, 'gpdcluster')
            id = id + 1
            if id%100 == 0 and id != 0:
                logger.debug("{} image descriptors were inserted so far.".format(id))
                print("{} image descriptors were inserted so far.".format(id))
    print("Inserted %d image descriptors."%id)
    print("Inserting image descriptors done.")


def createIndex(es, dim, mode):
    if mode == _METHODS_INS[0]:
        varname = "gpdcluster"
    else:
        varname = "gpd"

    mapping = {
        "mappings": {
            "properties": {
                "imageid": {
                    "type": "text"
                },
                "score": {
                    "type": "float"
                },
                varname: {
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

def insertdoc(es, gpdcluster, metadata, id, featurelabel):
    doc = {
    'imageid': metadata['image_id'],
    'score': metadata['score'],
    featurelabel: list(gpdcluster)
    }

    res = es.index(index=_INDEX, id=id, body=doc) 
    #print("inserting ",doc)
    if res['result'] != 'created':
        raise ValueError("Document not created.")

def query(es, featurevector, size, method):
    if method == _METHODS_SEARCH[0]:
        request = { "size": size,
                               "min_score": _SIMILARITY_TRESH + 1, 
                               "query": {
                                "script_score": {
                                "query": {
                                    "match_all": {}
                                },
                                "script": {
                                    "source": "cosineSimilarity(params.queryVector, doc['gpdcluster'])+1.0",
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
                           "source": "l2norm(params.queryVector, doc['gpd'])",
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

    logger.debug("Query returned {} results stated.".format(res['hits']['total']['value'])) 
    logger.debug("Query returned {} results actual.".format(len(res['hits']['hits']))) 
    
    print("Query returned {} results stated.".format(res['hits']['total']['value']))
    print("Query returned {} results actual.".format(len(res['hits']['hits'])))
    docs = res['hits']['hits']
    imageids = [item['_source']['imageid'] for item in docs]
    scores = [item['_score'] for item in docs] 
     
    return imageids, scores


def bestmatching_sumdist(image_scoring, k):
    #Scoring should contain for each imageid the accumulated scores between query features and db entries  
    score_sums = {}
    #Number of iterations: number of query features * number of db entries (each-by-each)
    start_time = time.time()
    print("Ranking query results with sum distance ...")
    for item in image_scoring:
        imageid = item[0] 
        score = item[1]
        if not imageid in score_sums:
            score_sums[imageid] = score
        else:
            score_sums[imageid] = score_sums[imageid] + score
    print("Ranking query results done. Took %s seconds."%(time.time() - start_time))
    print(score_sums)
    print("Max: ", max(score_sums.values()))
    print("Min: ", min(score_sums.values()))


    bestk = sorted(score_sums.items(), key=lambda x: x[1])[:k]
    #Apply normalization to intervall [0,1] & then apply exponential function, used for later comparison score in search results display
    exp_norm = lambda x: (x[0], np.exp( -10*(x[1] - min(score_sums.values()))/(max(score_sums.values()) - min(score_sums.values())) ))
    bestk = map(exp_norm, bestk)
    #print(list(bestk))
    #exp_norm = lambda x: (x[0], (x[1] - min(score_sums.values()))/(max(score_sums.values()) - min(score_sums.values()))) 
    #bestk = map(exp_norm, bestk)
    #print(list(bestk))
    #imageids = [x[0] for x in bestk]
    return list(bestk)


def bestmatching_cluster(image_scoring):
    grouped_by_imageid = [list(g) for k, g in itertools.groupby(sorted(image_scoring, key=lambda x:x[0]), lambda x: x[0])]
    
    #Format: [ [(img_id1,score1), (img_id1,score2)], ... [(img_idN,score1), .., (img_idN,scoreK)] ]
    #Rank by #occurances * mean of scores 
    start_time = time.time()
    print("Ranking query results with custom heuristic ...")
    ranked = sorted(grouped_by_imageid, key=lambda e: len(e) * np.mean([s[1] for s in e]), reverse=True)
    print("Ranking query results done. Took %s seconds."%(time.time() - start_time))
    
    #Only image ids as result
    ranked_reduced = []
    scoring = []
    for occurances in ranked:
        scores = [item[1] for item in occurances]
        score = sum(scores)/len(scores)
        ranked_reduced.append(occurances[0][0])
      
        scoring.append(score)
    
    #linear norm because already previous filtered with threshold
    print(scoring)
    minscore, maxscore = min(scoring), max(scoring)
    #all scores have the same value, normalize to all 1's
    if minscore == maxscore:
        scoring = [1 for i,x in enumerate(scoring)]
    else:
        lin_norm = lambda x: ( (x - minscore)/(maxscore - minscore) )
        scoring = map(lin_norm, scoring)
    
    bestk = list(zip(ranked_reduced, scoring))
    print("ranked:",bestk)
    return bestk


def get_alldocs(es):
    es.indices.refresh(index=_INDEX)
    response = scan(es, index=_INDEX, query={"query": { "match_all" : {}}})
    #for item in response:
    #    print(item)
    #print("------------------------")
    #print(list(response))
    #print("Number of documents in the database: ",len(list(response)))
    #exit(1)
    return list(response)
   
    #res = es.search(index=_INDEX, body={"query": {"match": { "match_all" : {}}} })
    #res = es.search(index=_INDEX, body={"query": {"match_all": {}}})
    
    """
    # Init scroll by search
    res = es.search(index=_INDEX, scroll='2m', size=100, body={})

    # Get the scroll ID
    sid = res['_scroll_id']
    scroll_size = len(res['hits']['hits'])
    print(res)

    while scroll_size > 0:
        data = res['hits']['hits']
        print("data: ",data)
        res = es.scroll(scroll_id=sid, scroll='2m')
        sid = res['_scroll_id']
        scroll_size = len(res['hits']['hits'])
    """

def saveResults(image_ids, rel_scores, output_dir, image_dir):
    imagemetadata = {'imagedir': image_dir}
    if rel_scores is None:
        for rank, imageid in enumerate(image_ids):
            imgid = str(imageid) 
            imgpath = "%s_%s.jpg"%( imgid[:len(imgid)-6].zfill(12), imgid[len(imgid)-6:])
            imgpath = os.path.join(image_dir, imgpath)
            imagemetadata[rank] = {'filepath': imgpath}
    else:
        for rank, (imageid, relscore) in enumerate(zip(image_ids, rel_scores)):
            imgid = str(imageid) 
            imgpath = "%s_%s.jpg"%( imgid[:len(imgid)-6].zfill(12), imgid[len(imgid)-6:])
            imagemetadata[rank] = {'filepath': imgpath, 'relscore': relscore}
    
    json_file = 'result-ranking'
    with open(os.path.join(output_dir, json_file+'.json'), 'w') as f:
        print("Writing to file: ",os.path.join(output_dir,json_file+'.json'))
        json.dump(imagemetadata, f, indent=4, separators=(',', ': '))
    print("Wrote %d ranked items to file."%len(image_ids))


if __name__=="__main__":
    main()

