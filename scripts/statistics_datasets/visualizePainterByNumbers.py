import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import re


pbn = pd.read_csv('datasetCSVs/painterbynumbers/all_data_info.csv')

#STYLE HISTOGRAM
a4_dims = (30, 12)
fig, ax = plt.subplots(figsize=a4_dims)
ax = sns.countplot(x='style', data=pbn, order =pbn['style'].value_counts().index)
ax.set(xlabel='Artistic Style', ylabel='Number of Artworks')
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
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

#ax.set_xticklabels(fontsize=7)
#ax.figure.autofmt_xdate()
plt.xticks(fontsize=7)
ax.figure.savefig('style_distr.png',bbox_inches='tight')

#Count number of different labels
print(pbn["style"].value_counts().index.tolist())
print("Number of style labels: " + str(len(pbn["style"].value_counts().index.tolist())) )


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
ax2.figure.savefig('year_distr.png',bbox_inches='tight')

#ORGIN HISTOGRAM
fig3, ax3 = plt.subplots()

ax3 = sns.countplot(x='source', data=pbn)
ax3.set(xlabel='Source Dataset', ylabel='Number of Artworks Used')
ax3.figure.savefig('origin_distr.png',bbox_inches='tight')

#GENRE STATISTICS


#DATASET STATISTICS
print("Number of images in train + test set: ",pbn.shape[0])
pbn_train = pd.read_csv('datasetCSVs/painterbynumbers/train_info.csv')
print("Number of images in train set: ",pbn_train.shape[0])