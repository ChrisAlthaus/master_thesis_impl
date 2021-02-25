from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import json
import random
import os

#Get named entities from dataset metadata

def get_continuous_chunks(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    continuous_chunk = []
    current_chunk = []
    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        if current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue
    #print("c:", continuous_chunk)
    return continuous_chunk


pbn = pd.read_csv('/home/althausc/master_thesis_impl/scripts/statistics_datasets/datasetCSVs/painterbynumbers/all_data_info.csv')
pbn = pbn[pbn['in_train'] == True]

titles = pbn['title'].to_list()
imageids = pbn['new_filename'].to_list()
namedentities_to_imgid = defaultdict(list)
namedentities_to_titles = defaultdict(list)
outputdir = '/home/althausc/master_thesis_impl/scripts/statistics_datasets/.stats/painterbynumbers'

for k, title in tqdm(enumerate(titles)):
    titlesentence = str(title) # + '.'
    namedentities = get_continuous_chunks(titlesentence)
    for ne in namedentities:
        namedentities_to_imgid[ne].append(imageids[k])
        namedentities_to_titles[ne].append(title)

ne_frequencies = [(ne,len(imgids)) for ne,imgids in namedentities_to_imgid.items()]
ne_frequencies = np.array(sorted(ne_frequencies, key=lambda x:x[1], reverse=True))

min_title_words = [1,2,3,4]
for minwords in min_title_words:
    ne_frequencies_subset = np.array([(ne,numimgids) for ne,numimgids in ne_frequencies if len(list(filter(lambda x: x!='.', word_tokenize(ne))))>= minwords])
    df = pd.DataFrame([ne_frequencies_subset[:,0], ne_frequencies_subset[:,1]], index=['named entity', 'imgid count']).T
    print(df[:4])

    ax = sns.barplot(data=df[:50], y='named entity', x='imgid count', orient='h')
    ax.set( xlabel='Frequency', ylabel='Named Entity')
    plt.yticks(fontsize=5)
    plt.xticks(fontsize=7)
    ax.figure.savefig(os.path.join(outputdir, 'named_entities_frequencies-%d.png'%minwords),bbox_inches='tight', dpi=300)
    plt.cla()

with open(os.path.join(outputdir, 'named-entities-to-imagids.json'), 'w') as f:
    json.dump(namedentities_to_imgid, f, indent=2)

min_title_words = [1,2,3,4]
for minwords in min_title_words:
    ne_ranked = {}
    for i,(ne, freq) in enumerate(ne_frequencies):
        titels = namedentities_to_titles[ne]
        if len(list(filter(lambda x: x!='.', word_tokenize(ne))))>=minwords:
            ne_ranked[i] = {ne: freq, 'titles': random.sample(titels, min(7,len(titels)))}
    with open(os.path.join(outputdir, 'named-entities-ranks-%d.json'%minwords), 'w') as f:
        json.dump(ne_ranked, f, indent=2)



    
