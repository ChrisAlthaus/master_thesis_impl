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
import csv

from utils import recursive_print_dict, logscorestats
from utils import checkbyrecalculate

import logging

#Setup of elasticsearch server and performing read/write on database indices.
#Different input data types are supported:
#   -GPD type: geometric pose descriptor type
#   -Inserting: either raw features or clusters
#   -Search method: either cossim based for cluster db or distance measure for raw features

#Similarity measures used by ranking procedure:
#   --> general idea: group retrieved results for all input descriptors by imageid 
#   1. Cluster database: > idea: suppress noise of individual feature vectors by cluster centers
#                        > chunk wise retrieval and then flatteing for all imgids
#                        > Number of result pairs (imgid, cos-score) = #input descriptors * _ELEMNUM_COS
#                        > ranking of imgids by: count(imgid)*meanscore(cossim-value-imgid)
#                        > linear norm to [0,1] for better differentiation, since often very similar ranking score
#   2. Raw feature database: > idea: for each input descriptor get _ELEMNUM_QUERYRESULT closest imgid entries, then take sum over all retrieval scores
#                            > ranking of imgids by: sum(L2-distance-imgid) 
#                            > Number of result pairs (imgid, l2score) = #input descriptors * _ELEMNUM_QUERYRESULT #TODO: Implement cossim
#                            > linear norm to [0,1] and then exponential norm


parser = argparse.ArgumentParser()
parser.add_argument('-file',type=str,
                    help='Json file with clustering centroids as keys and a list of image metadata as values.\
                    Or for search a dict of image descriptors')
parser.add_argument('-insert_data','-insert', action="store_true",
                    help='Whether to build an index from the raw descriptors or the cluster mapped data descriptors.')
parser.add_argument('-search_data','-search', action="store_true",
                    help='Whether to search the index for the input cluster data.')
parser.add_argument('-method_insert', help='Select method for insert data.')
parser.add_argument('-imgdir', help='Image directory the descriptors refer to (insert or search).')
parser.add_argument('-gpd_type', help='Select type of GPD descriptors.')
parser.add_argument('-method_search', help='Select method for searching data.')
parser.add_argument('-rankingtype', help='Select method for ranking found descriptors.')
parser.add_argument('-tresh', type=float, help='Similarity treshold for cossim result ranking.')
parser.add_argument('-delindex', type=str, help='Delete an index by name.')
parser.add_argument('-indexstats', type=str, help='Selects an index for debugging.')
parser.add_argument('-firstn', type=int, default=10, help='Gets firstn documents of the given index.')

args = parser.parse_args()

logger = logging.getLogger('elasticsearch-db')
_DEBUG = False#True#False#True#False #True #False
if _DEBUG:
    logger.setLevel(logging.DEBUG)
    # Setup ElasticSearch server & client logging
    logging.basicConfig()
    logging.getLogger('elasticsearch').setLevel(logging.DEBUG)
    logging.getLogger('urllib3').setLevel(logging.DEBUG)

#Min score used for cossim #deprecated?
if args.tresh is None:
    _SIMILARITY_TRESH = 0.95
else:
    _SIMILARITY_TRESH = args.tresh 
#Method 1 uses Cosinus-Similarity for comparing features with features in db produced by visual codebook preprocessing, 
#       just a shortlist of N best-matching documents is queried for better performance
#Method 2 uses a Distance-Measure to compute the sum of L2-distances between input features 
#       and raw features stored in the db (each similarity counts). The image with smallest distance is selected
_METHODS_INS = ['CLUSTER', 'RAW']
_METHODS_SEARCH = ['COSSIM', 'L1', 'L2']
_GPD_TYPES = ['JcJLdLLa_reduced', 'JLd_all'] #just used for index naming
#For multiple search descriptor always rankingtype querymuliple* will be selected 
_RANKING_TYPES = ['average', 'max']

_INDEX = 'cossim-test' #debug
if args.method_insert and args.gpd_type:
    _INDEX = 'imgid_gpd_%s_%s'%(args.method_insert, args.gpd_type)
    _INDEX = _INDEX +'_pbn10k'
    _INDEX = _INDEX.lower()
    _INDEX = 'patchesindexm10' 
    _INDEX = 'patchesindexm7'   
    _INDEX = 'patchesindexm5'    
    #_INDEX = 'bitseq4' 
if args.method_search:
    _INDEX = 'imgid_gpd_raw_jcjldlla_reduced_pbn10k'
    _INDEX = 'patchesindexm10' 
    _INDEX = 'patchesindexm7'
    _INDEX = 'patchesindexm5'    
    #_INDEX = 'bitseq4'   
#_INDEX = 'bitseq4' 
#_INDEX = '2vecstest' 
_INDEX = 'imgid_gpd_raw_jcjldlla_reduced_pbn10k'
print("Current index: ",_INDEX)

_ELEMNUM_QUERYRESULT = 1000 #bigger because relative score computation
_NUMRES = 100 #adjust for only take topk 

_CONFIGDIR = '/home/althausc/master_thesis_impl/retrieval/out/configs'

if args.method_search:
    if args.method_search not in _METHODS_SEARCH:
        raise ValueError("Please specify a valid search method.") 
if args.method_insert:
    if args.method_insert not in _METHODS_INS:
        raise ValueError("Please specify a valid insert method.")
if args.gpd_type:
    assert args.gpd_type in _GPD_TYPES
if args.rankingtype:
    assert args.rankingtype in _RANKING_TYPES


def main(): 
    data = None
    if args.insert_data or args.search_data:
        print("Reading from file: ",args.file)
        with open (args.file, "r") as f:
            data = f.read()
        data = eval(data)  
    print(data)
    output_dir = None
    if args.search_data:
        output_dir = os.path.join('/home/althausc/master_thesis_impl/retrieval/out/humanposes', datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            raise ValueError("Output directory %s already exists."%output_dir)

    
    es = Elasticsearch("http://localhost:9200", 
                       ca_certs=False,
                       verify_certs=False)
    _INDICES_ALL = es.indices.get_alias("*").keys()
    print("Existing Indices: ", _INDICES_ALL)
    
    if args.indexstats:
        global _INDEX
        _INDEX = args.indexstats
        show_docs(es, args.firstn)
        return

    _SRCIMG_DIR = getimgdir(_INDEX, imgdir=args.imgdir, es=es)
    print("Source image folder: ",_SRCIMG_DIR)

    
    if args.insert_data:
        if args.method_insert == _METHODS_INS[0]:
            createIndex(es, len(list(data.keys())[0]), _METHODS_INS[0], _SRCIMG_DIR)
            insertdata_cluster(args, data, es)
        elif args.method_insert == _METHODS_INS[1]:
            print(data[0])
            createIndex(es, len(data[0]['gpd']), _METHODS_INS[1], _SRCIMG_DIR)
            insertdata_raw(args, data, es)

        saveconfiginsert(_INDEX, len(data), args.file, _SRCIMG_DIR)

    elif args.search_data:
        assert _INDEX in _INDICES_ALL, "Index %s not found."%_INDEX
        es.indices.refresh(index=_INDEX)
        #data format: {'1': [featurevector], ... , 'n': [featurevector]}
        imgids_final = []
        
        results = []
        numElemAll = es.cat.count(_INDEX, params={"format": "json"})[0]['count']
        print("Total documents in the index: %d."%int(numElemAll))
        print("Searching image descriptors from %s ..."%args.file)
        #Number of result pairs (imgid, l2score) = #input descriptors * _ELEMNUM_QUERYRESULT
        for k,img_descriptor in enumerate(data):
            image_ids, scores = query(es, img_descriptor, _ELEMNUM_QUERYRESULT, args.method_search)
            results.append(list(zip(image_ids,scores)))
            print("Number of results for GPD {} = {}".format(k,len(image_ids)))
        print("Searching image descriptors done.")
        
        querynums = len(data)
        print("Results unranked: ", results)
        #exit(1)
        imgids_final = bestmatching(results, args.rankingtype, len(data), _NUMRES)
        #imgids_final = bestmatching_sumdist(results, _NUMRES)

        """elif args.method_search == _METHODS_SEARCH[2]:
            results = []
            print("Searching image descriptors from %s ..."%args.file)
            #Number of result pairs (imgid, l2score) = #input descriptors * _ELEMNUM_COS
            for img_descriptor in data:
                print(img_descriptor)
                image_ids, scores = query(es, img_descriptor, _ELEMNUM_COS, args.method_search)
                results.append(list(zip(image_ids,scores)))
            print("Searching image descriptors done.")
            print("QUERY RESULTS: ",results)
        
            #Flattening list of indiv feature vector results
            results = [item for sublist in results for item in sublist]
            imgids_final = bestmatching_cluster(results, _NUMRES)"""
        
        print("Best matched images: ", imgids_final)

        saveconfigsearch(output_dir, args, _INDEX, len(imgids_final), len(data))

        if isinstance(imgids_final[0], tuple):
            [image_ids, rel_scores] = zip(*imgids_final)
            saveResults(list(image_ids), list(rel_scores), output_dir, _SRCIMG_DIR)
        elif isinstance(imgids_final[0], int):
            saveResults(imgids_final, None, output_dir, _SRCIMG_DIR)

    elif args.delindex is not None:
        print("Deleting index with name: '%s' ..."%args.delindex)
        res = es.indices.delete(index=args.delindex, ignore=[400, 404])
        if 'error' in res:
            print('Error: ',res)
        else:
            print("Deleting index with name: %s done."%args.delindex)

def saveconfiginsert(indexname, numdocs, descriptorfile, imgdir):
    if not os.path.exists(os.path.join(_CONFIGDIR, 'elastic_config.csv')):
        with open(os.path.join(_CONFIGDIR, 'elastic_config.csv'), 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            headers = ['Indexname', 'Number Documents', 'Descriptor File', 'Src Image Folder']
            writer.writerow(headers)
            print("Wrote to config file: %s"%os.path.join(_CONFIGDIR, 'elastic_config.csv'))
    
    with open(os.path.join(_CONFIGDIR, 'elastic_config.csv'), 'a') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([indexname, numdocs, descriptorfile, imgdir])

def saveconfigsearch(output_dir, args, searchindex, numresults, numdescriptors):
     with open(os.path.join(output_dir, 'config.txt'), 'a') as f:
            f.write("Search Index: %s"%searchindex + os.linesep)
            f.write("GPD File: %s"%args.file + os.linesep)
            f.write("Search Method: %s"%args.method_search + os.linesep)
            f.write("Ranking Type: %s"%args.rankingtype + os.linesep)
            f.write("Number of Search Descriptors: %d"%numdescriptors + os.linesep)
            f.write("Number of Results: %d"%numresults + os.linesep)
            

def get_indexconfigs(indexname):
    with open(os.path.join(_CONFIGDIR, 'elastic_config.csv')) as f:
        cf = csv.reader(f, delimiter='\t')
        for row in reversed(list(cf)): #assumption: later entries are the newest/maybe replaced indices
            if len(row) == 0:
                return -1
            if row[0] == indexname:
                numdocs, descriptorfile, srcimg_dir = row[1], row[2], row[3]
                return [numdocs, descriptorfile, srcimg_dir]
        return -1


def insertdata_raw(args, data ,es):
    #Database layout:
    # 1.Raw features
    #   id | imagid | score |feature(gpd) 
    # 2.Cluster features
    #   id | imagid | score |feature(gpd cluster)            
    id = 0
    print("Inserting image descriptors from %s ..."%args.file)
    for item in data:
        d = {'gpd': item['gpd'], 'mask': item['mask']}
        metadata ={'image_id': item['image_id'], 'score': item['score']}
        insertdoc(es, d, metadata, id, 'gpd')
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
            insertdoc(es, gpdcluster, metadata, id, 'gpdcluster')   #TODO: error? items?
            id = id + 1
            if id%100 == 0 and id != 0:
                logger.debug("{} image descriptors were inserted so far.".format(id))
                print("{} image descriptors were inserted so far.".format(id))
    print("Inserted %d image descriptors."%id)
    print("Inserting image descriptors done.")


def createIndex(es, dim, mode, imgdir):
    if mode == _METHODS_INS[0]:
        varname = "gpdcluster"
    else:
        varname = "gpd"

    mapping = {
        "mappings": {
            "_meta": { 
                "imagedir": imgdir
            },
            "_source": {
                "enabled": True
            },
            "properties": {
                "imageid": {
                    "type": "text"
                },
                 "imid": {
                    "type": "integer"
                },
                "score": {
                    "type": "float"
                },
                varname: {
                    "type": "dense_vector",
                    "dims": dim
                },
                #Used to get the descriptor values in the script
                '{}-array'.format(varname): {
                    "type": "double" #"half_float"
                },
                'tempvec': {
                    "type": "double" #"half_float"
                },
                "mask": {
                    "type" : "keyword"#,
                    #"index" : False
                }    
            }               
        }
    }

    if es.indices.exists(index=_INDEX):
        print("Deleting existing index {}".format(_INDEX))
        es.indices.delete(index=_INDEX, ignore=[400, 404])

    response = es.indices.create(
        index=_INDEX,
        body=mapping,
        #ignore=400 # ignore 400 already exists code
    )
    if response['acknowledged'] != True:
        raise ValueError('Index was not created')
    print("Successfully created index: %s"%_INDEX)

def insertdoc(es, data, metadata, id, featurelabel):
    doc = {
    'imageid': metadata['image_id'],
    'imid': int(metadata['image_id']),
    'score': metadata['score'],
    featurelabel: list(data[featurelabel]),
    '{}-array'.format(featurelabel): list(data[featurelabel]),
    'mask': data['mask'],
    'tempvec':  list(np.random.randint(2, size=(4,))) #[0,0,0,0] #list(data[featurelabel]) # [-100 for i in range(len(data[featurelabel]))]
    }

    res = es.index(index=_INDEX, id=id, body=doc) 
    if res['result'] != 'created':
        raise ValueError("Document not created.")

def query(es, descriptor, size, method):
    featurevector = descriptor['gpd']
    maskstr = descriptor['mask']
    
    #Notes to querying in elasticsearch
    #   - params._source['gpd-array'] prevents es to sort the array (vs. params['gpd-array'])
    #   - cosineSimilarity(x,y) is only defined for y being a dense vector
    #       -> dense_vector cannot be modified or accessed in the script directly (not supported yet by es)
    #           ->can only be used as input to vectorscore functions
    #           ->therefore to mask/don't count -1 entries in the descriptors, the queryvector indices which shall be masked are set to the value of the dense_vector 
    #           ->for this, the array copy of the dense_vector is used to access the items

    #Metrics:
    #   1. Cos-Similarity Score: higher value means closer/better match (unlike cossim defined normaly)!
    print("QUERY:", featurevector)
    if method == _METHODS_SEARCH[0]: #COSSIM

        request = { "size": size,
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "lang":"painless",
                        "source": """
                            def m1 = params._source['mask'];
                            def m2 = params.queryMask;
                            double penalty = 0.5; //not necessary?!
                            double c = 0.0;

                            ArrayList qeffective = new ArrayList();
                            for(int i; i < m1.length(); i++) {
                                if (m1.charAt(i) == '0'.charAt(0) || m2.charAt(i) == '0'.charAt(0)) {
                                    qeffective.add(params._source['gpd-array'][i]);
                                    c = c + penalty;
                                }else{
                                    qeffective.add(params.queryVector[i]);
                                }
                            }
                            
                            double qdotproduct = 0;
                            double gpddotproduct = 0;
                            double qgpddotproduct = 0;

                            for (int dim = 0; dim < doc['gpd-array'].size(); dim++){
                                qdotproduct += qeffective[dim] * qeffective[dim];
                                gpddotproduct += params._source['gpd-array'][dim] * params._source['gpd-array'][dim];
                                qgpddotproduct += qeffective[dim] * params._source['gpd-array'][dim];
                            }

                            double qmagnitude = Math.sqrt(qdotproduct);
                            double gpdmagnitude = Math.sqrt(gpddotproduct);

                            double cossim = qgpddotproduct / (qmagnitude * gpdmagnitude);
                            return (cossim + 1) * params._source['score'];  
                            //return cosineSimilarity(qeffective, doc['gpd']) + 1 //not working
                            //return 1 / (1 + l1norm + c) *  params._source['score'];
                            """
                        ,
                        "params": {
                            "queryVector": list(featurevector),
                            "queryMask": maskstr
                        }   
                    }
                }
            }
          }

    elif method == _METHODS_SEARCH[1]: #L1-distance

        request = { "size": size,
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "lang":"painless",
                        "source": """
                            def m1 = params._source['mask'];
                            def m2 = params.queryMask;
                            double penalty = 0.5; //0.5 since majority of entries are between [0,1] & Manhatten distance
                            double c = 0.0;

                            ArrayList qeffective = new ArrayList();
                            for(int i; i < m1.length(); i++) {
                                if (m1.charAt(i) == '0'.charAt(0) || m2.charAt(i) == '0'.charAt(0)) {
                                    qeffective.add(params._source['gpd-array'][i]);
                                    c = c + penalty;
                                }else{
                                    qeffective.add(params.queryVector[i]);
                                }
                            }
                            
                            double l1norm = 0;
                            for (int dim = 0; dim < doc['gpd-array'].size(); dim++){
                                l1norm += Math.abs(params._source['gpd-array'][dim] - qeffective[dim]);
                            }
                            //return l1norm;
                            return 1 / (1 + l1norm + c) *  params._source['score'];
                            //return 1 / (1 + l1norm(qeffective, doc['gpd']) + c) *  params._source['score']; //not working
                            """
                        ,
                        "params": {
                            "queryVector": list(featurevector),
                            "queryMask": maskstr
                        }   
                    }
                }
            }
          }
                  #Debug descriptors: - insert: /home/althausc/master_thesis_impl/posedescriptors/out/query/11-25_18-18-11/geometric_pose_descriptor_c_1_mJcJLdLLa_reduced_t0.05_f1_mkpt10n1.json
                  #             -query: /home/althausc/master_thesis_impl/posedescriptors/out/query/11-13_13-35-19/geometric_pose_descriptor_c_1_mJcJLdLLa_reduced_t0.05_f1_mkpt10n1.json
                  
    elif method == _METHODS_SEARCH[2]: #L2-distance
         request = { "size": size,
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "lang":"painless",
                        "source": """
                            def m1 = params._source['mask'];
                            def m2 = params.queryMask;
                            double penalty = 0.5/params.queryVector.size(); //0.5 since majority of entries are between [0,1]
                            double c = 0.0;

                            ArrayList qeffective = new ArrayList();
                            for(int i; i < m1.length(); i++) {
                                if (m1.charAt(i) == '0'.charAt(0) || m2.charAt(i) == '0'.charAt(0)) {
                                    qeffective.add(params._source['gpd-array'][i]);
                                    c = c + penalty;
                                }else{
                                    qeffective.add(params.queryVector[i]);
                                }
                            }

                            double l2norm = 0;
                            for (int dim = 0; dim < doc['gpd-array'].size(); dim++){
                                double diff = params._source['gpd-array'][dim] - qeffective[dim];
                                l2norm += diff * diff;
                            }
                            l2norm = Math.sqrt(l2norm);
                            return 1 / (1 + l2norm + c) *  params._source['score'];
                            //return 1 / (1 + l2norm) *  params._source['score'];
                            //return l2norm(qeffective, doc['gpd']) *  params._source['score']; not working
                            """
                        ,
                        "params": {
                            "queryVector": list(featurevector),
                            "queryMask": maskstr
                        }   
                    }
                }
            }
          }
    else:
        raise ValueError()
    try :
        res= es.search(index=_INDEX, 
                        body=request, explain=True, search_type='dfs_query_then_fetch')
    except elasticsearch.ElasticsearchException as es1:  
        print("Error when querying for feature vector ",featurevector)
       
        print(es1)
        print("---------------------------------")
        print(es1.info)
        print("---------------------------------")
        recursive_print_dict(es1.info)
        print("---------------------------------")

    if _DEBUG:
        print("Query returned {} results stated.".format(res['hits']['total']['value']))
        print("Query returned {} results actual.".format(len(res['hits']['hits'])))
        print("-------------------------------------")
        #print("Raw result output \n:", res)
        #print("-------------------------------------")
        recursive_print_dict(res)
        docs = res['hits']['hits']
        imageids = [item['_source']['imageid'] for item in docs]
        scores = [item['_score'] for item in docs]
        print("Query descriptor: ", descriptor)
        print("Result: ",imageids ,scores)
        resultlist = list(zip(imageids, scores))
        for id, s in resultlist:
            print(id, s)
        checkbyrecalculate(resultlist, docs, method , featurevector, maskstr)
    
    docs = res['hits']['hits']
    imageids = [item['_source']['imageid'] for item in docs]
    scores = [item['_score'] for item in docs] 

    return imageids, scores

def bestmatching(image_scoring, rankingtype, querynums, k):
    #Scoring should contain for each imageid the accumulated scores between query features and db entries  
    score_sums = {}
    #Number of iterations: number of query features * number of db entries (each-by-each)
    start_time = time.time()

    c_gpds = 0
    if querynums>1:
        rankingtype = 'querymultiple-firstn' #'querymultiple-average'
        print("Choosing rankingtpye = {}, because {} search descriptors.".format(rankingtype, querynums))
        #Only take best matched pose for each image(-id) for each query descriptor
        #To prevent to ranking images high with only 1 query descriptor seeing mulitple times
        #Each query descriptor should be have equal influence on ranking
        r_bestperid = []
        for qresult in image_scoring:
            groupbyid = itertools.groupby(sorted(qresult, key=lambda x:x[0]), lambda x: x[0])
            for id, group in groupbyid:
                scores = [s for id,s in group]
                r_bestperid.append((id, max(scores)))
                c_gpds = c_gpds + len(scores)
        image_scoring = r_bestperid
        print("Reduced,because of multiple query descriptors: \n {} -> {} ".format(c_gpds, len(image_scoring)))

    else:
        #Flattening list of indiv feature vector results
        image_scoring = [item for sublist in image_scoring for item in sublist]
        c_gpds = len(image_scoring)
    
    print("Ranking query results with average score over unique imageids ...")
    image_scoring = sorted(image_scoring, key=lambda x:x[0])
    grouped_by_imageid = itertools.groupby(image_scoring, lambda x: x[0])
    #for k,v in grouped_by_imageid:
    #    print(k, v,list(v))
    #exit(1)
    print("Using ranking type: {}".format(rankingtype))
    
    c_ids = 0
    for imageid, group in grouped_by_imageid:
        if rankingtype == 'average':
            score_sums[imageid] = np.mean([s for id,s in group])#TODO: single/multiple pose switch? maybe
        elif rankingtype == 'max':
            score_sums[imageid] = np.max([s for id,s in group])
        elif rankingtype == 'querymultiple-firstn':
            scores = [s for id,s in group]
            #print(scores)
            #Get up to number of query descriptor search results
            scores = scores[:querynums]
            scores.sort(reverse=True)
            score_sums[imageid] = scores[0] + sum([s/querynums for s in scores[1:]]) if len(scores)>1 else 0
            #print(score_sums[imageid])
            #print("----------------------------")
        elif rankingtype == 'querymultiple-average':
            score_sums[imageid] = np.mean([s for id,s in group])
        else:
            raise ValueError()
        c_ids = c_ids + 1
    
    print("Ranking query results done. Took %s seconds."%(time.time() - start_time))
    assert c_ids == len(score_sums)
    print("Number of processed search result gpds: ", c_gpds)
    print("Number of unique imagids: ",len(score_sums))
    print("Every imgid has in average {} returned descriptors".format(c_gpds/len(score_sums)))
    print("Raw averagescore statistics:", logscorestats(list(score_sums.values())))

    bestk = sorted(score_sums.items(), key=lambda x: x[1], reverse=True)[:k]
   
    if not _DEBUG:
        #Apply normalization to intervall [0,1] & then apply exponential function, used for later comparison score in search results display
        if max(score_sums.values()) != min(score_sums.values()):
            lin_norm = lambda x: (x[0], (x[1] - min(score_sums.values()))/(max(score_sums.values()) - min(score_sums.values())))
            bestk = list(map(lin_norm, bestk))
            print(bestk)

            print("Linear Normalization statistics:", logscorestats([s for id,s in bestk]))
    #log_norm = lambda x: (x[0], np.log2(x[1]+1)) 
    #bestk = list(map(exp_norm, bestk))
    #print("Logarithmic Normalization statistics:", logscorestats([s for id,s in bestk]))
    print("Best k: ", bestk)
    return bestk


#deprecated
def bestmatching_cluster(image_scoring, k):
    grouped_by_imageid = [list(g) for k, g in itertools.groupby(sorted(image_scoring, key=lambda x:x[0]), lambda x: x[0])]
    print("Raw group meanscore statistics:", logscorestats([np.mean([s[1] for s in e]) for e in grouped_by_imageid]))
    print("Raw group size statistics:", logscorestats([len(e) for e in grouped_by_imageid]))

    #Format: [ [(img_id1,score1), (img_id1,score2)], ... [(img_idN,score1), .., (img_idN,scoreK)] ]
    #Rank by (#occurances above Q3 treshold) * mean of scores 
    #Use Q3 treshold above all groups to allow for boost of very certain matches.
    #Otherwise entries with low score value will falsify boost factor.
    print("Ranking query results with custom heuristic ...")
    q3_tresh = np.mean([np.percentile(np.asarray(e[1]), [75]) for e in grouped_by_imageid])
    ranked = sorted(grouped_by_imageid, key=lambda e: max(1, len([s[1] for s in e if s[1] > q3_tresh])) * np.mean([s[1] for s in e]), reverse=True)
    print("Ranking query results done. Took %s seconds."%(time.time() - start_time))
    
    #Only image ids & updated scores as result
    #Resulting scores (=mean) may not be in decreasing order since ordering rule: (#occurances above Q3 treshold) * mean of scores #deprectaed
    imageids = []
    scoring = []
    for chunkid in ranked:
        imageids.append(chunkid[0][0])
        scoring.append( max(1, len([s[1] for s in chunkid if s[1] > q3_tresh])) * np.mean([s[1] for s in chunkid]) )
        
    print("Ranked score statistics (by ordering rule):", logscorestats(scoring))

    #linear norm for better differentiation & search browsing
    if max(scoring) != min(scoring):
        print("Applying linear norm.")
        lin_norm = lambda x: ( (x - min(scoring))/(max(scoring) - min(scoring)) )
        scoring = map(lin_norm, scoring)
    
    bestk = list(zip(imageids, scoring))[:k]
    return bestk

def show_docs(es, firstn):
    print("Getting first {} documents of index {} ...".format(firstn, _INDEX))
    es.indices.refresh(index=_INDEX)
    response = scan(es, index=_INDEX, query={"query": { "match_all" : {}}}, size=firstn)

    recursive_print_dict({'Documents':list(response)})
    #print(list(response)[:firstn])

def get_alldocs(es):
    es.indices.refresh(index=_INDEX)
    response = scan(es, index=_INDEX, query={"query": { "match_all" : {}}})

    return list(response)
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

def getimgdir(indexname, imgdir=None, es=None):
    #Options to specify/infer the image directory:
    # 1. specify by arguments
    # 2. if not, then get from local config
    # 3. if no entry for the index, then search on es server                 
    if imgdir is not None:
        if os.path.isdir(imgdir):
            print('Getting index configuration from arguments.')
            return imgdir
        else:
            raise ValueError("No valid src image directory.")
    else:
        index_config = get_indexconfigs(indexname)
        if index_config!= -1:
            numdocs, descriptorfile, srcimg_dir = index_config
            print('Getting index configuration from config file.')
        else:
            srcimg_dir = es.indices.get_mapping(_INDEX)[_INDEX]['mappings']['_meta']['imagedir']
            print('Getting index configuration from server.')
        return srcimg_dir


def saveResults(image_ids, rel_scores, output_dir, image_dir):
    #Output file format:
    #   - imagdir: base image directory
    #   - dict-entries rank : {filepath, relscore}
    imagemetadata = {'imagedir': image_dir}
    if rel_scores is None:
        rel_scores = ['unknown'] * len(image_ids)

    for rank, (imageid, relscore) in enumerate(zip(image_ids, rel_scores)):
        imgid = str(imageid)
        root, ext = os.path.splitext(imgid)
        if not ext:
            imgname = "%s.jpg"%imgid
        else:
            imgname = imgid
        imagemetadata[rank] = {'filepath': imgname, 'relscore': relscore}
    
    json_file = 'result-ranking'
    with open(os.path.join(output_dir, json_file+'.json'), 'w') as f:
        print("Writing to file: ",os.path.join(output_dir,json_file+'.json'))
        json.dump(imagemetadata, f, indent=4, separators=(',', ': '))
    print("Wrote %d ranked items to file."%len(image_ids))


if __name__=="__main__":
    main()

