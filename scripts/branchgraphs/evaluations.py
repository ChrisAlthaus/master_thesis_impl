import os
import json 
import argparse
import numpy as np
import math
import pickle
import datetime
import time
import logging
import itertools

parser = argparse.ArgumentParser()
args = parser.parse_args()

def latestdir(dir):
    diritems = [os.path.join(dir, d) for d in os.listdir(dir)]
    all_subdirs = [d for d in diritems if os.path.isdir(d)]
    return max(all_subdirs, key=os.path.getmtime)

def filewithname(dir, searchstr):
    for item in os.listdir(dir):
        if os.path.isfile(os.path.join(dir,item)) and searchstr in item:
            return os.path.join(dir,item)
    return None


# ---------------------------- PREDICTION STATISTICS ------------------------------
print("PREDICTION STATISTICS:")
predfile = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/out/predictions/graphs/09-28_12-23-05/custom_prediction.json'

cmd = "python3.6 /home/althausc/master_thesis_impl/scripts/scenegraph/utils/graph_statistics.py -graphfile {}"\
                                                .format(predfile)
print(cmd)

outrun_dir = os.path.join(os.path.dirname(predfile), '.stats')
print("Output Directory: %s\n"%outrun_dir)

