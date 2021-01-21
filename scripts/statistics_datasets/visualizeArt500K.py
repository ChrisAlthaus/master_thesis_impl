import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os

import re
import json
import csv

#Read headers, append one column because of missing header
headers = None
with open('/nfs/data/iart/art500k/img/head_info.csv', newline='') as f:
  reader = csv.reader(f)
  headers = next(reader)  # gets the first line
  print(headers)

with open('/nfs/data/iart/art500k/img/label_list.csv', newline='') as f:
  reader = csv.reader(f)
  row1 = next(reader)  # gets the first line
  print(row1)

headers[-1] = 'Unknown'
headers.append('Path')
dtype={}
for i in range(len(headers)):
    headers[i] = headers[i].replace(" ", "")
    headers[i] = headers[i].replace("'", "")
    dtype[headers[i]] = str
print(headers)

#Read labeled data
art500klabels = pd.read_csv('/nfs/data/iart/art500k/img/label_list.csv', delimiter='|', header=None, names=headers, dtype=dtype)
print(art500klabels)
print("First N Paths:")
print(art500klabels['Path'][:10])

for imgname in art500klabels['Path'].values:
    if 'qwFguWgqijxC4A.jpg' in str(imgname):
      print(imgname)
print("Search done.")
exit(1)

photographs = ['Amnesty##IwHYC2Srhy7Zjw', 'Churchill At M I T##4wGb-TDGBXusXg', 'NoName##zgHoHf19xrBr2Q', 'Modern Playground Equipment##zQGa146h5N_h9Q']
for photoname in photographs:
    print(art500klabels.loc[art500klabels['painting_name'] == photoname].values)


interestheaders = ['Genre', 'Style', 'Nationality', 'Painting School', 'Art Movement', 'Field', 'Media', 'Influenced on', 'Art institution', 'Period', 'Theme']
interestheaders = headers
for h in interestheaders:
    print(h, len(art500klabels[h].unique()))
    print(art500klabels[h].unique()[:10])
    print()

photomedia = []
for m in art500klabels['Media']:
    if isinstance(m,str):
        if 'photo' in m:
            photomedia.append(m)
print(photomedia)
print(len(photomedia))

#print([x for x in art500klabels[" 'Media'"] if 'photo' in x])


print("Number of row with field column == painting:")
print(len(art500klabels.loc[art500klabels['Field'] == 'painting']))

outputdir = '/home/althausc/master_thesis_impl/scripts/statistics_datasets/.stats/art500k'
a4_dims = (50, 22)
fig, ax = plt.subplots(figsize=a4_dims)
ax = sns.countplot(data=art500klabels, y='Field', order =art500klabels['Field'].value_counts().index[:150], color='tab:blue')
ax.set( xlabel='Number of Artworks', ylabel='Artistic Field')

plt.xticks(fontsize=7)
ax.figure.savefig(os.path.join(outputdir, 'field_histogram.png'),bbox_inches='tight')

#Check images of some field categories
print("Field == Print:")
print(art500klabels.loc[art500klabels['Field'] == 'Print']['image_url'].values[:10])
print("Field == Street Art:")
print(art500klabels.loc[art500klabels['Field'] == 'street art']['image_url'].values[:10])
print(art500klabels[:10].values)

#Search for some images, to test special character paths
print("1:", art500klabels.loc[art500klabels['painting_name'] == '108/Bombe 400Ml Customis\udcc3\udca9e##NwFFcx2DikiVpg.jpg'])
print("2:", art500klabels.loc[art500klabels['Path'] == 'Artists1/Drawing/Mæss Mulgyul 1211##XAFNp3gpZfXkGw.jpg'])

print(art500klabels.loc[art500klabels['painting_name'].str.contains('qwFguWgqijxC4A.jpg')])
