import os
import json 
import argparse
import numpy as np
import math
import datetime
import time
import logging
from collections import OrderedDict, defaultdict

from PIL import Image

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def main():
    ne_to_imgids_filepath = '/home/althausc/master_thesis_impl/scripts/statistics_datasets/.stats/painterbynumbers/named-entities-to-imagids.json'

    print("Reading from file: ",ne_to_imgids_filepath)
    with open(ne_to_imgids_filepath, "r") as f:
        ne_to_imageids = json.load(f)

    ne = "Saint Anthony" #"Crucifixion" #"Madonna" #"Saint Anthony" #"Temptation of Christ" #Temptation of St. Anthony" #'Leda'
    nes = ["Saint Anthony" ,"Crucifixion" ,"Madonna" ,"Saint Anthony", "Temptation of Christ", "Temptation of St. Anthony" ,"Leda"]
    nes = ["David", "Aphrodite", "Adoration"]

    for ne in nes:
        imageids = ne_to_imageids[ne]
        savepath = os.path.join('/home/althausc/master_thesis_impl/scripts/statistics_datasets/.stats/painterbynumbers/namedentitiyimages', '{}.jpg'.format(ne)) 

        nrows = min(8, math.ceil(len(imageids)/8))
        ncolumns = 8
        imgdir = '/nfs/data/iart/kaggle/img'
        title = 'Image which title contains NE \'{}\' ({} images)'.format(ne, len(imageids))
        plotImageGrid(imageids[:nrows*ncolumns], title, nrows, ncolumns, imgdir, savepath)



def plotImageGrid(imgids, title, nrows, ncolumns, imgdir, savepath):
    fig = plt.figure(figsize=(ncolumns/2, nrows/2+1))
    fig.suptitle(title, y=0.9, fontsize=5)

    fig.tight_layout()
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(nrows, ncolumns),  # creates 2x2 grid of axes
                    axes_pad=0.05,  # pad between axes in inch.
                    share_all=True
                    )   

    images = []
    for filename in imgids:
        filepath = os.path.join(imgdir, filename)
        img = Image.open(filepath)
        img = resizeimage(img)
        images.append(np.array(img))
    
    c_added = 0
    for ax, im in zip(grid, images):
        # Iterating over the grid returns the Axes.
        ax.axis('off')
        ax.imshow(im)
        #imtitle = 'r={} d={:0.2f}'.format(rid, float(cdist))
        #ax.text(0.5,-0.1, imtitle, size=2, ha="center", transform=ax.transAxes)
        c_added = c_added + 1
    
    if c_added<nrows*ncolumns:
        for i in range(c_added,nrows*ncolumns):
            grid[i].axis('off')

    plt.savefig(savepath, dpi=400, bbox_inches='tight', pad_inches=0.01)
    plt.clf()
    plt.cla()

def resizeimage(image):
    MAX_SIZE = 400
    original_size = max(image.size[0], image.size[1])
    if original_size >= MAX_SIZE:
        #if (image.size[0] > image.size[1]):
        resized_width = MAX_SIZE
        resized_height = int(round((MAX_SIZE/float(image.size[0]))*image.size[1])) 
        #else:
        #    resized_height = MAX_SIZE
        #    resized_width = int(round((MAX_SIZE/float(image.size[1]))*image.size[0]))
        image = image.resize((resized_width, resized_height), Image.ANTIALIAS)
    return image

if __name__=="__main__":
    main()