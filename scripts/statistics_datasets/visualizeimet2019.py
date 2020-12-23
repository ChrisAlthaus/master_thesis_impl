import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os
import csv

import re
import json

outputdir = '/home/althausc/master_thesis_impl/scripts/statistics_datasets/.stats/imet2019'
if not os.path.exists(outputdir):
    os.makedirs(outputdir)

data = []
with open('/home/althausc/master_thesis_impl/scripts/statistics_datasets/train.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    for row in csvreader:
        data.append(row)

print(data[:2])

imet = pd.read_csv('/home/althausc/master_thesis_impl/scripts/statistics_datasets/train.csv')
#print(imet[:100])

labels = pd.read_csv('/nfs/data/iart/imet_2019/labels.csv')
#print(labels[:100])

attr_id_to_name = {}
categories = []
for index, row in labels.iterrows():
    category, name =  row['attribute_name'].split('::')
    categories.append(category)
    attr_id_to_name[row['attribute_id']] = {'category':category, 'name':name}

print("Number of unique categories: ", len(set(categories)))
assert len(set(categories)) == 2

c_attribute_culture = defaultdict(int)
c_attribute_tag = defaultdict(int)

for index, row in imet.iterrows():
    attrids =  map(int, row['attribute_ids'].split(' '))
    for id in attrids:
        category = attr_id_to_name[id]['category']
        attrname = attr_id_to_name[id]['name']

        if category == 'culture':
            c_attribute_culture[attrname] += 1
        elif category == 'tag':
            c_attribute_tag[attrname] += 1
        else:
            raise ValueError()
print(list(c_attribute_culture.items())[:10])

print("Number of unique culture attr: ", len(set(list(c_attribute_culture.keys()))))
print("Number of unique tag attr: ", len(set(list(c_attribute_tag.keys()))))

attr_c1 = np.array(sorted(c_attribute_culture.items(), key=lambda x:x[1], reverse=True))
srclabels_c1 = ['imet'] * len(attr_c1)

df = pd.DataFrame([attr_c1[:,0], srclabels_c1, attr_c1[:,1]], index=['attrname', 'src', 'counts']).T
a4_dims = (30, 12)
fig, ax = plt.subplots(figsize=a4_dims)
ax = sns.barplot(data=df , y='attrname', x='counts', hue='src', order=df['attrname'][:50], orient='h')
ax.set( xlabel='Number of Artworks', ylabel='Culture Attributes')   
plt.xticks(fontsize=7)
ax.figure.savefig(os.path.join(outputdir, 'culture_attributes.png'),bbox_inches='tight')
plt.cla()


attr_c2 = np.array(sorted(c_attribute_tag.items(), key=lambda x:x[1], reverse=True))
srclabels_c2 = ['imet'] * len(attr_c2)

df = pd.DataFrame([attr_c2[:,0], srclabels_c2, attr_c2[:,1]], index=['attrname', 'src', 'counts']).T
a4_dims = (30, 12)
fig, ax = plt.subplots(figsize=a4_dims)
ax = sns.barplot(data=df , y='attrname', x='counts', hue='src', order=df['attrname'][:50], orient='h')
ax.set( xlabel='Number of Artworks', ylabel='Tag Attributes')   
plt.xticks(fontsize=7)
ax.figure.savefig(os.path.join(outputdir, 'tag_attributes.png'),bbox_inches='tight')
plt.cla()

c_attr_ordered = [{attr:c} for attr,c in attr_c2]
with open(os.path.join(outputdir, 'tag-attributes-count.json'), 'w') as f:
    json.dump(c_attr_ordered, f, indent=2)

