import json
import csv
import os
import pandas as pd
from PIL import Image
import ast

#deprecated: not used

# -------------------------------------------------------------------------

headers = None
with open('/nfs/data/iart/art500k/img/head_info.csv', newline='') as f:
  reader = csv.reader(f)
  headers = next(reader)  # gets the first line

headers[-1] = 'Unknown'
headers.append('Path')

dtype={}
for i in range(len(headers)):
    headers[i] = headers[i].replace(" ", "")
    headers[i] = headers[i].replace("'", "")
    dtype[headers[i]] = str
print(headers)

art500klabels = pd.read_csv('/nfs/data/iart/art500k/img/label_list.csv', delimiter='|', header=None, names=headers, dtype=dtype)
specialchars = 'ÆÐƎƏƐƔĲŊŒẞÞǷȜæð'
specialchars = [c for c in specialchars]

#if 'Artists1/João Baptista Da Costa/Quaresmas##FAE4E-xDMeeRzA.jpg' in art500klabels['Path'].values:
#        print("found")
#
#labels = []
#for l in art500klabels['Path'].values:
#    labels.append(str(l).encode('utf-8'))
#
#with open('/home/althausc/master_thesis_impl/detectron2/out/art_predictions/train/01-21_18-09-11/filenames.json', 'w') as f: #, encoding='utf8'
#    filenamesd = {index: path for index,path in enumerate(art500klabels['Path'].values)}
#    json.dump(filenamesd, f) #, ensure_ascii=False)
#with open ('/home/althausc/master_thesis_impl/detectron2/out/art_predictions/train/01-21_18-09-11/filenames.json', "r") as f:
#    filenamesd = json.load(f)
#
##json_labels = json.load(art500klabels['Path'].values)
#
#with open ('/home/althausc/master_thesis_impl/detectron2/out/art_predictions/train/01-21_18-09-11/maskrcnn_predictions.json', "r") as f:
#    json_data = json.load(f)
#for item in json_data:
#    #print(item['image_id'].encode('utf-8'))
#    imgpath = item['image_id']
#    #print(imgpath)
#    #print(art500klabels['Path'].values[:10])
#
#    
#    if imgpath in filenamesd.values():
#        #print(imgpath)
#        print("contained")
#    else:
#        #print(imgpath)
#        print("not contained")
#       
#    #img = Image.open(imgpath)
#
#exit(1)

"""
Artists1/Claude Monet/Regnvær Etretat##hAGxTh_ktg_DlQ.jpg
Artists1/Post-Impressionism/Trær Og Hus Provence##IAFWl3-k88JZ3g.jpg
Artists1/Modern art/Trær Og Hus Provence##IAFWl3-k88JZ3g.jpg
Artists1/Drawing/Mæss Mulgyul 1211##XAFNp3gpZfXkGw.jpg
"""
print("Number of images in the dataset: ", len(art500klabels))
fieldallowed = ['painting', 'Painting', 'sketch', 'Sketch', 'print', 'Print', 'drawing', 'Drawing', 'Sculpture', 'sculpture', 'street art']
artwork_paths = []
for field in fieldallowed:
    subframe = art500klabels[art500klabels['Field'].str.contains(field, na=False)]
    artpaths = subframe['Path'].values.astype(str)
    artpaths = [os.path.basename(str(filepath)) for filepath in artpaths]
    artwork_paths.extend(artpaths)
    #for i in subframe['Path'].values.astype(str):
    #    for c in specialchars:
    #        if c in i:
    #            print(i)

print("Reduced artworks: ", len(artwork_paths))
artwork_paths.sort()
print(artwork_paths[:10])

predictionfiles = ['/home/althausc/master_thesis_impl/detectron2/out/art_predictions/train/01-20_18-06-22/maskrcnn_predictions.json']

for pfile in predictionfiles:
    with open (pfile, "r") as f:
            json_data = json.load(f)
     
    json_data = sorted(json_data, key=lambda k: k['image_id'])
    for item in json_data:
        if 'FAFB_kIxpRxHXQ' in item['image_id']:
            print(item['image_id'])
    
    exit(1)
    #for item in json_data[5:10]:
    #    print(item['image_id'])
    #    print(os.path.basename(item['image_id']))
    
    rpredictions = []
    uniqueids1 = []
    uniqueids2 = []
    uniqueids21 = []
    notfound = set()
    print("Number of predictions: ", len(json_data))
    for k,item in enumerate(json_data, start=1):
        imageid = os.path.basename(str(item['image_id']))
        if imageid in artwork_paths:
            rpredictions.append(item)
            if imageid not in uniqueids2:
                uniqueids2.append(item['image_id'])
                uniqueids21.append(imageid)
        else:
            notfound.add(imageid)
        if imageid not in uniqueids1:
            uniqueids1.append(imageid)

        if k%10000 == 0:
            print("Processed {} image ids".format(k))

    print("Number of unique image ids previous: ",len(uniqueids1))
    print("Number of unique image ids now: ",len(uniqueids2), len(uniqueids21))
    print(notfound)

    foldername = os.path.dirname(pfile)
    with open(os.path.join(foldername, 'maskrcnn_predictions_nophotographs.json'), 'w') as f:
        json.dump(rpredictions, f)

    with open(os.path.join(foldername, 'config_postprocessing.txt'), 'a') as f:
            f.write("Number of predictions: {}".format(len(json_data)) + os.linesep)
            f.write("Number of unique image ids previous: {}".format(len(uniqueids1)) + os.linesep)
            f.write("Number of unique image ids nows: {}".format(len(uniqueids2)) + os.linesep)

