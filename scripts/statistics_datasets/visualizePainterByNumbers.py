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

pbn = pd.read_csv('/home/althausc/master_thesis_impl/scripts/statistics_datasets/datasetCSVs/painterbynumbers/all_data_info.csv')
pbn = pbn[pbn['in_train'] == True]

gpdfile = '/home/althausc/master_thesis_impl/posedescriptors/out/insert/12-18_16-44-58/geometric_pose_descriptor_c_53615_mJcJLdLLa_reduced_t0.05_f1_mkpt7n1.json'
with open (gpdfile, "r") as f:
    data = f.read()
    dbdata = eval(data)
imgids = list(set([d["image_id"] for d in dbdata]))

dbmetadata = []
for index, row in pbn.iterrows():
    #print(int(os.path.splitext(row['filename'])[0]), imgids[:4])
    #exit(1)
    if os.path.splitext(row['new_filename'])[0] in imgids: 
        dbmetadata.append(row)
dbmetadata = pd.DataFrame(dbmetadata)

print('Number of images in kaggle pbn: ',len(pbn))
pbn_train = pd.read_csv('datasetCSVs/painterbynumbers/train_info.csv')
print("Number of images in train set (for sanity check): ", pbn_train.shape[0])
print('Number of images in database: ',len(dbmetadata))


def getmcountsdf(attrname, pbn, dbmetadata):
    attr_c1 = defaultdict(int)
    for index, row in pbn.iterrows():
        attr_c1[row[attrname]] += 1

    attr_c2 = {key:0 for key in attr_c1.keys()}
    for index, row in dbmetadata.iterrows():
        attr_c2[row[attrname]] += 1

    attr_c1 = np.array(sorted(attr_c1.items(), key=lambda x:x[1], reverse=True))
    srclabels_c1 = ['pbn'] * len(attr_c1)
    attr_c2 = np.array(sorted(attr_c2.items(), key=lambda x:x[1], reverse=True))
    srclabels_c2 = ['db'] * len(attr_c2)

    genres = np.concatenate((attr_c1[:,0], attr_c2[:,0]))
    counts = np.concatenate((attr_c1[:,1], attr_c2[:,1]))
    srclabels = srclabels_c1 + srclabels_c2

    df = pd.DataFrame([genres, srclabels, counts], index=[attrname, 'src', 'counts']).T
    return df

def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

# -------------------------------- GENRE HISTOGRAMS -----------------------------------------------
#GENRE HISTOGRAM
a4_dims = (30, 12)
fig, ax = plt.subplots(figsize=a4_dims)
ax = sns.countplot(data=pbn, y='genre', order =pbn['genre'].value_counts().index)
ax.set(xlabel='Number of Artworks', ylabel='Artistic Genre')
#sns.set(rc={'figure.figsize':(11.7,8.27)})
#change_width(ax, 1)
#ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
#ax.set_xticklabels(fontsize=7)
#ax.figure.autofmt_xdate()
plt.xticks(fontsize=7)
ax.figure.savefig(os.path.join(outputdir, 'genre_distr_pbn.png'),bbox_inches='tight')

#Count number of different labels
print("Number of genre labels: " + str(len(pbn["genre"].value_counts().index.tolist())) )
plt.cla()
plt.clf()

#GENRE HISTOGRAM REDUCED
a4_dims = (30, 12)
fig, ax = plt.subplots(figsize=a4_dims)
ax = sns.countplot(data=dbmetadata, y='genre', order =dbmetadata['genre'].value_counts().index)
ax.set( xlabel='Number of Artworks', ylabel='Artistic Genre')

plt.xticks(fontsize=7)
ax.figure.savefig(os.path.join(outputdir, 'genre_distr_reduced_pbn.png'),bbox_inches='tight')
plt.cla()

#Count number of different labels
print("Number of genre labels: " + str(len(dbmetadata["genre"].value_counts().index.tolist())) )

#GENRE HISTOGRAMS BOTH
#Get genre distributions merged
genredf = getmcountsdf('genre', pbn, dbmetadata)
a4_dims = (30, 12)
fig, ax = plt.subplots(figsize=a4_dims)
ax = sns.barplot(data=genredf, y='genre', x='counts', hue='src', orient='h')
ax.set( xlabel='Number of Artworks', ylabel='Artistic Genre')

#ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.xticks(fontsize=7)
ax.figure.savefig(os.path.join(outputdir, 'genre_distr_comparison.png'),bbox_inches='tight')
plt.cla()
plt.clf()

# -------------------------------- STYLE HISTOGRAMS -----------------------------------------------
firstnstyles = 30
#GENRE HISTOGRAM REDUCED
a4_dims = (30, 12)
fig, ax = plt.subplots(figsize=a4_dims)
ax = sns.countplot(data=dbmetadata, y='style', order =dbmetadata['style'].value_counts().index[:firstnstyles])
ax.set( xlabel='Number of Artworks', ylabel='Artistic Style')

plt.xticks(fontsize=7)
ax.figure.savefig(os.path.join(outputdir, 'style_distr_reduced_pbn.png'),bbox_inches='tight')
plt.cla()

#STYLE HISTOGRAMS BOTH
#Get style distributions merged
styledf = getmcountsdf('style', pbn, dbmetadata)
a4_dims = (30, 12)
fig, ax = plt.subplots(figsize=a4_dims)
ax = sns.barplot(data=styledf, y='style', x='counts', hue='src', order=styledf['style'][:firstnstyles], orient='h')
ax.set( xlabel='Number of Artworks', ylabel='Artistic Style')

#ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.xticks(fontsize=7)
ax.figure.savefig(os.path.join(outputdir, 'style_distr_comparison.png'),bbox_inches='tight')
plt.cla()

# -------------------------------- YEAR HISTOGRAMS -----------------------------------------------
#YEAR DISTRIBUTION
dims = (15, 12)
fig2, ax2 = plt.subplots(figsize=dims)
#print(pbn["date"].value_counts())
#print(type(pbn["date"].value_counts()))
#print(pbn["date"].value_counts().values)
#print(pbn["date"].value_counts().keys().size)

def getyeardistribution(df):
    series_year = df["date"].value_counts()
    c_yearsartworks = defaultdict(int)

    for index, val in series_year.iteritems():
        pattern = re.compile('.*?(\d{1,4})')
        match = pattern.match(index)
        if match is None:
            print("Year not found in string: ",index)
        else:
            i = int(match.group(1))
            c_yearsartworks[i] += val
            if(i<1000):
                print("Year smaller than 1000")
                print(i,val)
    #print(len(set(xs)), len(xs))
    #assert len(set(xs)) == len(xs), 'Multiple years entry'
    xs = list(c_yearsartworks.keys())
    ys = list(c_yearsartworks.values())
    return xs,ys

#series_year = pd.Series(ys, index=xs).sort_index()
#ax2 = series_year.hist(bins=100)
xs,ys = getyeardistribution(pbn)
df_year_pbn = pd.DataFrame({'Year':xs, 'Number of Artworks':ys})
df_year_pbn.sort_values(by=['Year'], ascending=False)
ax2 = sns.lineplot(x='Year',y='Number of Artworks', data=df_year_pbn, ci=None)
ax2.figure.savefig(os.path.join(outputdir, 'year_distr_pbn.png'),bbox_inches='tight')
plt.cla()

xs,ys = getyeardistribution(dbmetadata)
df_year_db = pd.DataFrame({'Year':xs, 'Number of Artworks':ys})
df_year_db.sort_values(by=['Year'], ascending=False)
ax2 = sns.lineplot(x='Year',y='Number of Artworks', data=df_year_db, ci=None)
ax2.figure.savefig(os.path.join(outputdir, 'year_distr_reduced_pbn.png'), bbox_inches='tight')
plt.cla()

df_year_pbn.loc[:,'source'] = 'pbn'
df_year_db.loc[:,'source'] = 'db'

merged = pd.concat([df_year_pbn, df_year_db])
ax2 = sns.lineplot(x="Year", y="Number of Artworks", hue="source", data=merged)
ax2.figure.savefig(os.path.join(outputdir, 'year_distr_both.png'), bbox_inches='tight')
plt.cla()


# -------------------------------- ARTISTS HISTOGRAMS -----------------------------------------------
firstnartists = 70
#ARTIST HISTOGRAM REDUCED
a4_dims = (30, 12)
fig, ax = plt.subplots(figsize=a4_dims)
ax = sns.countplot(data=dbmetadata, y='artist', order =dbmetadata['artist'].value_counts().index[:firstnartists])
ax.set( xlabel='Number of Artworks', ylabel='Artist')

plt.xticks(fontsize=7)
ax.figure.savefig(os.path.join(outputdir, 'artist_distr_reduced_pbn.png'),bbox_inches='tight')

#ARTIST HISTOGRAM BOTH
artistsdf = getmcountsdf('artist', pbn, dbmetadata)
ax = sns.barplot(x="counts", y="artist", hue="src", data=artistsdf, order=artistsdf['artist'][:firstnartists], orient='h')
ax.set( xlabel='Number of Artworks', ylabel='Artist')
ax.figure.savefig(os.path.join(outputdir, 'artists_distr_both.png'),bbox_inches='tight')
plt.cla()
plt.clf()

# -------------------------------- ORIGIN HISTOGRAMS -----------------------------------------------
#ORGIN HISTOGRAM
fig3, ax3 = plt.subplots()

ax3 = sns.countplot(x='source', data=pbn)
ax3.set(xlabel='Source Dataset', ylabel='Number of Artworks Used')
ax3.figure.savefig(os.path.join(outputdir, 'origin_distr.png'),bbox_inches='tight')
plt.clf()
