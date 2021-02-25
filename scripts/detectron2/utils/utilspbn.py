import json
import os
import random

#Utility functions for the PaintersByNumbers dataset

def randomrankings(numberrankings, k, savedir):
    datasetfolder = '/nfs/data/iart/kaggle/img/'
    artworkpaths = [f for f in os.listdir(datasetfolder) if os.path.isfile(os.path.join(datasetfolder, f))]
    rankingpaths = []

    for i in range(numberrankings):
        imgpaths = random.sample(artworkpaths, k)
        rankingdict = {"imagedir": datasetfolder}
        for n,imgpath in enumerate(imgpaths):
            rankingdict[str(n)] = { "filename": imgpath, "relscore": 0.0}

        json_file = 'result-ranking-random-{}'.format(i)
        with open(os.path.join(savedir, json_file+'.json'), 'w') as f:
            print("Writing to file: ",os.path.join(savedir,json_file+'.json'))
            json.dump(rankingdict, f, indent=4, separators=(',', ': '))

        rankingpaths.append(os.path.join(savedir, json_file+'.json'))
    return rankingpaths