import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os

import re
import json
import csv

headers = None
with open('/nfs/data/iart/art500k/img/head_info.csv', newline='') as f:
  reader = csv.reader(f)
  headers = next(reader)  # gets the first line
  print(headers)
#headers = headers[1:]
headers.append("end")

with open('/nfs/data/iart/art500k/img/label_list.csv', newline='') as f:
  reader = csv.reader(f)
  row1 = next(reader)  # gets the first line
  print(row1)

dtype={}
for h in headers:
    dtype[h] = str

art500klabels = pd.read_csv('/nfs/data/iart/art500k/img/label_list.csv', delimiter='|', header=None, names=headers, dtype=dtype)
print(art500klabels)

interestheaders = [" 'Genre'", " 'Style'", " 'Nationality'", " 'Painting School'", " 'Art Movement'", " 'Field'", " 'Media'", " 'Influenced on'", " 'Art institution'", " 'Period'", " 'Theme'"]
interestheaders = headers
for h in interestheaders:
    print(h, len(art500klabels[h].unique()))
    print(art500klabels[h].unique()[:10])
    print()

photomedia = []
for m in art500klabels[" 'Media'"]:
    if isinstance(m,str):
        if 'photo' in m:
            photomedia.append(m)
print(photomedia)
print(len(photomedia))
#print([x for x in art500klabels[" 'Media'"] if 'photo' in x])