import json
import csv
import os
import pandas as pd
from PIL import Image
import ast
import random
import pathlib

def get_paths_of_paintings():
    #Get all painting related filepaths from the artwork500k csv file.
    #That is necessary because there are a lot of photographs and other non-relevant images in the dataset like fragments or vases.
    fieldallowed = ['painting', 'Painting', 'sketch', 'Sketch', 'print', 'Print', 'drawing', 'Drawing', 'Sculpture', 'sculpture', 'street art']
    foldernames = ['Artists1', 'Artists2', 'Art Movement', 'Medium']
    artwork_paths = []
    c = 0
    n_notfound = 0
    n_refound = 0

    # The dataset contains some paths which are not valid
    # This gives erros in prediction:
    # [Errno 2] No such file or directory: b'/nfs/data/iart/art500k/img/Artists1/Drawing/A Wooded Landscape Near Beekhuizen##ZQEtaY4yNy0KdA.jpg'
    # [Errno 2] No such file or directory: b'/nfs/data/iart/art500k/img/Artists1/Drawing/A Maja Seated On A Chair With A Male And Female Companion Folio 18 Verso From The Madrid Album B##ogHVQ9ouorptGA.jpg'
    # [Errno 2] No such file or directory: b'/nfs/data/iart/art500k/img/Artists1/Drawing/Design For Brougham No 3427##jAHANZ2So5yzRw.jpg'
    # [Errno 2] No such file or directory: b'/nfs/data/iart/art500k/img/Artists1/Drawing/Dwarf White Bauhinia Bauhinia Tomenlosa Linn##FQGzpdr7khfBTg.jpg'
    # Sometimes false base directory is used in .csv, therefore try other basedirectories for not found paths
    # e.g. Artists1/Drawing/A Wooded Landscape Near Beekhuizen##ZQEtaY4yNy0KdA.jpg -> Medium/Drawing/A Wooded Landscape Near Beekhuizen##ZQEtaY4yNy0KdA.jpg

    # Note:
    # When UnicodeError (happens more when running on GPU), the standard solution is to encode strings with .encode('utf-8') 

    print("Pre-Processing of filepaths from .csv ...")
    with open('/home/althausc/nfs/data/artimages/label_list.csv', 'r', encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file, delimiter='|')
        for row in reader:
            c+= 1
            if c%100000 == 0:
                print("Processed {} .csv file paths.".format(c))
            labelstr = ''.join([str(label) for label in row])
            for field in fieldallowed:
                if field in labelstr:
                    if '.jpg' in row[-1]:
                        filepath = ''
                        if os.path.isfile(os.path.join('/nfs/data/iart/art500k/img', row[-1]).encode('utf-8')):
                            filepath = os.path.join('/nfs/data/iart/art500k/img', row[-1])
                        else:
                            for foldern in foldernames:
                                fname_without_firstdir = os.path.join(*pathlib.Path(row[-1]).parts[1:])
                                fname = os.path.join(foldern, fname_without_firstdir)

                                if os.path.isfile(os.path.join('/nfs/data/iart/art500k/img', fname).encode('utf-8')):
                                    filepath = os.path.join('/nfs/data/iart/art500k/img', fname)
                                    n_refound += 1
                                    break  
                        if filepath:
                            artwork_paths.append(filepath)
                        else:
                            n_notfound += 1
                            #print("File {} not found, searched in all artwork directories.".format(row[-1].encode('utf-8'))) 
                    break
    print("Pre-Processing of filepaths from .csv done.")
    
    print("\nNumber of artworks in .csv:", c)
    print("Total Number of painting paths found:", len(artwork_paths))
    print("Number of paintings paths not found:", n_notfound)
    print("Number of paintings paths found in other base directories:", n_refound)
    
    print("\nRandom sample of painting paths:")
    for sample in random.sample(artwork_paths, 50):
        try:
            print(sample)
        except UnicodeEncodeError as e:
            print(sample.encode('utf-8'))

    return artwork_paths

def searchcsv(filestr):
    with open('/home/althausc/nfs/data/artimages/label_list.csv', 'r', encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file, delimiter='|')
        for row in reader:
            if filestr in row[-1]:
                print(row[-1])
    
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



if __name__ == "__main__":
    searchcsv('The Church St Sofia')  

    get_paths_of_paintings()      