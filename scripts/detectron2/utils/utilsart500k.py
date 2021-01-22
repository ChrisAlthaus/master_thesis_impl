import json
import csv
import os
import pandas as pd
from PIL import Image
import ast
import random

def get_paths_of_paintings():
    fieldallowed = ['painting', 'Painting', 'sketch', 'Sketch', 'print', 'Print', 'drawing', 'Drawing', 'Sculpture', 'sculpture', 'street art']
    artwork_paths = []
    c = 0
    with open('/home/althausc/nfs/data/artimages/label_list.csv', 'r', encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file, delimiter='|')
        for row in reader:
            c+= 1
            labelstr = ''.join([str(label) for label in row])
            for field in fieldallowed:
                if field in labelstr:
                    if '.jpg' in row[-1]:
                        artwork_paths.append(os.path.join('/nfs/data/iart/art500k/img', row[-1]))
                    break

    print("Number of artworks in .csv:", c)
    print("Number of painting paths:", len(artwork_paths))
    print("Random sample of painting paths:")
    for sample in random.sample(artwork_paths, 50):
        try:
            print(sample)
        except UnicodeEncodeError as e:
            print("Unicode cannot be printed: ",e)

    return artwork_paths

def testing():
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

    #Errors: loading error of some rows, since last two columns (Field and Path) are sometimes swapped
    art500klabels = pd.read_csv('/nfs/data/iart/art500k/img/label_list.csv', delimiter='|', header=None, names=headers, dtype=dtype)
    print(art500klabels)
    for items in art500klabels[:4].values:
        print(list(zip(headers,items)))
        
    print(art500klabels['Field'][:100])
    print("Number of images in the dataset: ", len(art500klabels))
    fieldallowed = ['painting', 'Painting', 'sketch', 'Sketch', 'print', 'Print', 'drawing', 'Drawing', 'Sculpture', 'sculpture', 'street art']
    artwork_paths = []
    for field in fieldallowed:
        subframe = art500klabels[art500klabels['Field'].str.contains(field, na=False)]
        artpaths = subframe['Path'].values.astype(str)

        nanrows = subframe.loc[subframe['Path'].isnull()]
        artpaths = [os.path.join('/nfs/data/iart/art500k', str(filepath)) for filepath in artpaths if str(filepath)!='nan']
        artwork_paths.extend(artpaths)

    print("Reduced artworks: ", len(artwork_paths))
    artwork_paths.sort()
    print(artpaths[:10])
    print(random.sample(artwork_paths, 50))