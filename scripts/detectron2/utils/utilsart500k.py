import json
import csv
import os
import pandas as pd
from PIL import Image
import ast


headers = None
with open('/nfs/data/iart/art500k/img/head_info.csv', newline='') as f:
  reader = csv.reader(f)
  headers = next(reader)  # gets the first line

headers[-1] = 'Path' #'Unknown'
#headers.append('Path')

dtype={}
for i in range(len(headers)):
    headers[i] = headers[i].replace(" ", "")
    headers[i] = headers[i].replace("'", "")
    headers[i] = headers[i].replace("[", "")
    dtype[headers[i]] = str
print(headers)

art500klabels = pd.read_csv('/nfs/data/iart/art500k/img/label_list.csv', delimiter='|', header=None, names=headers, dtype=dtype)
print(art500klabels)
for items in art500klabels[:4].values:
    print(list(zip(headers,items)))

fieldallowed = ['painting', 'Painting', 'sketch', 'Sketch', 'print', 'Print', 'drawing', 'Drawing', 'Sculpture', 'sculpture', 'street art']
artwork_paths = []
import csv
with open('/nfs/data/iart/art500k/img/label_list.csv','r') as csv_file:
    reader =csv.reader(csv_file, delimiter='|')
    for row in reader:
        labelstr = ''.join([str(label) for label in row])
        for field in fieldallowed:
            if field in labelstr:
                if '.jpg' in row[-1]:
                    artwork_paths.append(row[-1])
                break
print(len(artwork_paths))
import random
print(random.sample(artwork_paths, 50))
exit(1)

print(art500klabels['Field'][:100])
exit(1)
print("Number of images in the dataset: ", len(art500klabels))
fieldallowed = ['painting', 'Painting', 'sketch', 'Sketch', 'print', 'Print', 'drawing', 'Drawing', 'Sculpture', 'sculpture', 'street art']
artwork_paths = []
for field in fieldallowed:
    subframe = art500klabels[art500klabels['Field'].str.contains(field, na=False)]
    artpaths = subframe['Path'].values.astype(str)

    nanrows = subframe.loc[subframe['Path'].isnull()]
    print(nanrows[:10])
    exit(1)
    artpaths = [os.path.join('/nfs/data/iart/art500k', str(filepath)) for filepath in artpaths if str(filepath)!='nan']
    artwork_paths.extend(artpaths)

print("Reduced artworks: ", len(artwork_paths))
artwork_paths.sort()
print(artpaths[:10])
import random
print(random.sample(artwork_paths, 50))