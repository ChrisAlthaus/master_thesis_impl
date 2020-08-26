import datetime
from elasticsearch import Elasticsearch, RequestsHttpConnection
import argparse
import json
import ast

parser = argparse.ArgumentParser()
parser.add_argument('-clusterFile','-clustering',required=True,
                    help='Json file with clustering centroids as keys and a list of image metadata as values.')
parser.add_argument('-insert_data','-insert', action="store_true",
                    help='Whether to build an index from the cluster mapped data.')
parser.add_argument('-search_data','-search', action="store_true",
                    help='Whether to search the index for the input cluster data.')

args = parser.parse_args()

_INDEX = 'imgid_gpdcluster'

def main():
    print("Reading from file: ",args.clusterFile)
    with open (args.clusterFile, "r") as f:
        data = f.read()
    data = eval(data)

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
        createIndex(es)

        #data format: {gpdcluster1: [{img_id, score, vis}, ... ,{img_id, score, vis}], ... , gpdclusterK: [{img_id, score, vis}, ... ,{img_id, score, vis}]}
        id = 0
        for gpdcluster, imgs_metadata in data:
            for metadata in imgs_metadata:
                insertdoc(es, gpdcluster, metadata, id)
                id = id + 1
    elif args.search_data:
        #data format: {'1': [featurevector], ... , 'n': [featurevector]}
        results = []
        for i, feature_vector in data.item():
            image_ids = query(feature_vector)
            results.append(image_ids)


def createIndex(es):
    mapping = {
        "mappings": {
            "properties": {
                "imageid": {
                    "type": "integer"
                },
                "score": {
                    "type": "float"
                },
                "gpdcluster": {
                    "type": "dense_vector",
                    "dims": 4
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

def insertdoc(es, gpdcluster, metadata, id):
    doc = {
    'imageid': metadata['image_id'],
    'score': metadata['score'],
    'gpdcluster': list(gpdcluster)
    }

    res = es.index(index=_INDEX, id=id, body=doc)
    #print(res['result'])
    if not res['created']:
        raise ValueError("Document not created.")

def query(featurevector):
    return None
    
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


