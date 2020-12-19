import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os

import re
import json

outputdir = '/home/althausc/master_thesis_impl/scripts/statistics_datasets/.stats/painterbynumbers'
if not os.path.exists(outputdir):
        os.makedirs(outputdir)

pbn = pd.read_csv('/home/althausc/master_thesis_impl/scripts/statistics_datasets/datasetCSVs/painterbynumbers/train_info.csv')


gpdfile = '/home/althausc/master_thesis_impl/posedescriptors/out/insert/12-18_16-44-58/geometric_pose_descriptor_c_53615_mJcJLdLLa_reduced_t0.05_f1_mkpt7n1.json'
with open (gpdfile, "r") as f:
    data = f.read()
    dbdata = eval(data)
imgids = list(set([d["image_id"] for d in dbdata][:1000]))

dbmetadata = []
for index, row in pbn.iterrows():
    #print(int(os.path.splitext(row['filename'])[0]), imgids[:4])
    #exit(1)
    if os.path.splitext(row['filename'])[0] in imgids: 
        dbmetadata.append(row)
dbmetadata = pd.DataFrame(dbmetadata)

print('Number of images in kaggpe pbn: ',len(pbn))
print('Number of images in database: ',len(dbmetadata))

#GET STYLE DISTRIBUTIONS
genre_c1 = defaultdict(int)
for index, row in pbn.iterrows():
    genre_c1[row['genre']] += 1

genre_c2 = {key:0 for key in genre_c1.keys()}
for index, row in dbmetadata.iterrows():
    genre_c2[row['genre']] += 1

genre_c1 = sorted(genre_c1.items(), key=lambda x:x[1], reverse=True)
genre_c2 = [genre_c2[genre] for genre,count in genre_c1]
genres = [genre for genre,count in genre_c1]
genre_c1 = [count for genre,count in genre_c1]

genredf = pd.DataFrame([genres, genre_c1, genre_c2], index=['genre', 'pbn', 'db']).T

#STYLE HISTOGRAM
a4_dims = (30, 12)
fig, ax = plt.subplots(figsize=a4_dims)
ax = sns.countplot(data=pbn, y='genre', order =pbn['genre'].value_counts().index)
ax.set(xlabel='Number of Artworks', ylabel='Artistic Style')
#sns.set(rc={'figure.figsize':(11.7,8.27)})


def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)
#change_width(ax, 1)
#ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

#ax.set_xticklabels(fontsize=7)
#ax.figure.autofmt_xdate()
plt.xticks(fontsize=7)
ax.figure.savefig(os.path.join(outputdir, 'genre_distr_pbn.png'),bbox_inches='tight')

#Count number of different labels
print("Number of genre labels: " + str(len(pbn["genre"].value_counts().index.tolist())) )

#STYLE HISTOGRAM REDUCED
a4_dims = (30, 12)
fig, ax = plt.subplots(figsize=a4_dims)
ax = sns.countplot(data=dbmetadata, y='genre', order =dbmetadata['genre'].value_counts().index)
ax.set( xlabel='Number of Artworks', ylabel='Artistic Style')

#ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.xticks(fontsize=7)
ax.figure.savefig(os.path.join(outputdir, 'genre_distr_reduced_pbn.png'),bbox_inches='tight')

#Count number of different labels
print("Number of genre labels: " + str(len(dbmetadata["genre"].value_counts().index.tolist())) )

#STYLE HISTOGRAMS BOTH
a4_dims = (30, 12)
fig, ax = plt.subplots(figsize=a4_dims)
ax = sns.barplot(data=genredf, y='genre', x=['pbn', 'db'], orient='h')
ax.set( xlabel='Number of Artworks', ylabel='Artistic Style')

#ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.xticks(fontsize=7)
ax.figure.savefig(os.path.join(outputdir, 'genre_distr_comparison.png'),bbox_inches='tight')


#YEAR DISTRIBUTION
dims = (15, 12)
fig2, ax2 = plt.subplots(figsize=dims)
print(pbn["date"].value_counts())
print(type(pbn["date"].value_counts()))
print(pbn["date"].value_counts().values)
print(pbn["date"].value_counts().keys().size)

series_year = pbn["date"].value_counts()
xs = []
ys = []

for index, val in series_year.iteritems():
    pattern = re.compile('.*?(\d{1,4})')
    match = pattern.match(index)
    if match is None:
        print("Year not found in string: ",index)
    else:
        i = int(match.group(1))
        xs.append(i)
        ys.append(val)
        if(i<1000):
            print("Year smaller than 1000")
            print(i,val)

#series_year = pd.Series(ys, index=xs).sort_index()
#ax2 = series_year.hist(bins=100)
df_year = pd.DataFrame({'Year':xs, 'Number of Artworks':ys})
df_year.sort_values(by=['Year'], ascending=False)

ax2 = sns.lineplot(x='Year',y='Number of Artworks', data=df_year, ci=None)
ax2.figure.savefig(os.path.join(outputdir, 'year_distr.png'),bbox_inches='tight')

#ORGIN HISTOGRAM
fig3, ax3 = plt.subplots()

ax3 = sns.countplot(x='source', data=pbn)
ax3.set(xlabel='Source Dataset', ylabel='Number of Artworks Used')
ax3.figure.savefig(os.path.join(outputdir, 'origin_distr.png'),bbox_inches='tight')

#GENRE STATISTICS


#DATASET STATISTICS
print("Number of images in train + test set: ",pbn.shape[0])
pbn_train = pd.read_csv('datasetCSVs/painterbynumbers/train_info.csv')
print("Number of images in train set: ",pbn_train.shape[0])