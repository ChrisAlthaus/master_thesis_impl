import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


pbn = pd.read_csv('datasetCSVs/painterbynumbers/all_data_info.csv')

#print(pbn)
#ax = pbn['style'].value_counts().plot(kind='bar')
a4_dims = (30, 12)
fig, ax = plt.subplots(figsize=a4_dims)
ax = sns.countplot(ax=ax, x='style', data=pbn, order =pbn['style'].value_counts().index)
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
ax.figure.savefig('style_distr.png')

#Count number of different labels
print(pbn["style"].value_counts().index.tolist())
print("Number of style labels: " + str(len(pbn["style"].value_counts().index.tolist())) )

