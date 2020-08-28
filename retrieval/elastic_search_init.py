import datetime
from elasticsearch import Elasticsearch, RequestsHttpConnection
from elasticsearch.helpers import scan
import elasticsearch
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
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")

args = parser.parse_args()

logger = logging.getLogger('elasticsearch-db')
if args.verbose:
    logger.setLevel(logging.DEBUG)
#logging.basicConfig(level=logging.DEBUG, format='%(message)s')

_INDEX = 'imgid_gpdcluster'

def main():
    print("Reading from file: ",args.file)
    with open (args.file, "r") as f:
        data = f.read()
    data = eval(data)

    if args.insert_data:
        for i in range(len(data)-4):
            del data[list(data.keys())[0]]
        
        for gpd, items in data.items():
            for i in range(len(items)-4):
                del items[0]
        print(len(data))

        length = sum([len(buckets) for buckets in data.values()])
        print("Items in input data: ",length)
    elif args.search_data:
         print("Items in input data: ",len(data))
    #print(type(clustermap))
    #for key,value in clustermap.items():

    # you can use RFC-1738 to specify the url
    #es = Elasticsearch(['devbox3.research.tib.eu@localhost:443'])
    #es = Elasticsearch([{'host': 'althausc@devbox3.research.tib.eu', 'port': 9200}])
    #es = Elasticsearch('https://localhost:30000', 
    #                    verify_certs=False,
    #                    connection_class=RequestsHttpConnection)
    #es = Elasticsearch(['https://user:secret@localhost:30000'])

    #import ssl
    #ssl_defaults = ssl.get_default_verify_paths()
    #sslopt_ca_certs = {'ca_certs': ssl_defaults.cafile}

    es = Elasticsearch("http://localhost:30000",
                       ca_certs=False,
                       verify_certs=False)
    
    if args.insert_data:
        createIndex(es, len(list(data.keys())[0]))

        #data format: {gpdcluster1: [{img_id, score, vis}, ... ,{img_id, score, vis}], ... , gpdclusterK: [{img_id, score, vis}, ... ,{img_id, score, vis}]}
        id = 0
        print("Inserting image descriptors from %s ..."%args.file)
        for gpdcluster, imgs_metadata in data.items():
            for metadata in imgs_metadata:
                insertdoc(es, gpdcluster, metadata, id)
                id = id + 1
                if id%100 == 0 and id != 0:
                    logger.debug("{} image descriptors were inserted so far.".format(id))
                    print("{} image descriptors were inserted so far.".format(id))
        print("Inserting image descriptors done.")
        get_alldocs(es)

    elif args.search_data:
        #data format: {'1': [featurevector], ... , 'n': [featurevector]}
        results = []
        print("Searching image descriptors from %s ..."%args.file)
        for img_descriptor in data:
            print(img_descriptor)
            image_ids, scores = query(es, img_descriptor['gpd'])
            results.append(list(zip(image_ids,scores)))
        print("Searching image descriptors done.")
        print(results)


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
                "gpdcluster": {
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

def insertdoc(es, gpdcluster, metadata, id):
    doc = {
    'imageid': metadata['image_id'],
    'score': metadata['score'],
    'gpdcluster': list(gpdcluster)
    }

    res = es.index(index=_INDEX, id=id, body=doc) 
    #print("inserting ",doc)
    if res['result'] != 'created':
        raise ValueError("Document not created.")

def query(es, featurevector):
    try :
        res= es.search(index=_INDEX, 
                        body={  "query": {
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
                        })
    except elasticsearch.ElasticsearchException as es1:  
        print("Error when querying for feature vector "+featurevector)
        print(es1) 
    logger.debug("Query returned {} results.".format(res['hits']['total']['value'])) 
    print("Query returned {} results.".format(res['hits']['total']['value']))
    docs = res['hits']['hits']
    imageids = [item['_source']['imageid'] for item in docs]
    scores = [item['_source']['score'] for item in docs]  
    gpds =  [item['_source']['gpdcluster'] for item in docs]
    print("feature vector: ", featurevector)
    print("GPDS: ",gpds)     
    return imageids, scores

def get_alldocs(es):
    es.indices.refresh(index=_INDEX)
    response = scan(es, index=_INDEX, query={"query": { "match_all" : {}}})
    for item in response:
        print(item)
    #print(list(response))
    exit(1)
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

def test():  
    es = Elasticsearch("http://localhost:30000",
                       ca_certs=False,
                       verify_certs=False)



    doc = {
        'imageid': 39,
        'score': 0.98,
        'gpdcluster': [1,1,1,1]
    }
    res = es.index(index=_INDEX, id=1, body=doc)
    print(res['result'])
    print(res)

    doc = {
        'imageid': 40,
        'score': 0.76,
        'gpdcluster': [100,100,100,100]
    }
    res = es.index(index=_INDEX, id=2, body=doc)
    print(res['result'])

    res = es.get(index=_INDEX, id=1)
    print(res)
    print(res['_source'])


    es.indices.refresh(index=_INDEX)

    res= es.search(index=_INDEX,body={'query':{'match_all':{}}})
    print("search1: ",res)
    res= es.search(index=_INDEX, body={'query':{'match':{'imageid': 39}}})
    print("search2: ",res)
    res= es.search(index=_INDEX, body={'query':{'match':{'imageid': 40}}})
    print("search3: ",res)

    try :
        res= es.search(index=_INDEX, 
                        body={  "query": {
                                "script_score": {
                                "query": {
                                    "match_all": {}
                                },
                                "script": {
                                    "source": "cosineSimilarity(params.queryVector, doc['gpdcluster'])+1.0",
                                    "params": {
                                    "queryVector": [2,2,2,2]  
                                    }
                                }
                                }
                            }
                        })
    except elasticsearch.ElasticsearchException as es1:  
        print("Error")
        print(es1)              
    print(res)

    #res = es.search(index="test-index", body={"query": {"match_all": {}}})
    #print("Got %d Hits:" % res['hits']['total']['value'])
    #for hit in res['hits']['hits']:
    #    print("%(timestamp)s %(author)s: %(text)s" % hit["_source"])

if __name__=="__main__":
    main()
    #test()

