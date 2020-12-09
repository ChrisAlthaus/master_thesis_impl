import random
import numpy as np
import datetime
import json
import os
import time
import csv

inputfile = ' /home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/out/predictions/graphs/12-08_18-42-37/.descriptors/graphdescriptors.json'

epochs = 100
traineval_epoch = 10
valeval_epoch = 10
saveepoch = 100
valsize = 0.2
evaltopk = 100

dimensionsize = 1024 #1024 #32 #128
wliters = 3
downsampling = 0.0001

gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/sbatch_nogpu.sh'
jobname = 'graph2vec-train%s'%datetime.datetime.now().strftime('%d-%H-%M-%S')

logdir = '/home/althausc/master_thesis_impl/graph2vec/models/logs'
logfile = os.path.join(logdir, '%s.log'%datetime.datetime.now().strftime('%m-%d-%H-%M-%S'))

cmd =  ("sbatch -w devbox4 -J {} -o {} "+ \
            "{} python3.6 -u /home/althausc/master_thesis_impl/graph2vec/src/graph2vec.py "+ \
            "--input-path {} --output-path {} --workers 4 --dimensions {} --epochs {} --wl-iterations {} --down-sampling {} "+ \
			"--epochsave {} --traineval {} --valeval {} --valsize {} --evaltopk {}")\
				.format(jobname, logfile, gpu_cmd, inputfile, 'notused', dimensionsize, epochs, wliters, downsampling, 
                        saveepoch, traineval_epoch, valeval_epoch, valsize, evaltopk)
print(cmd)
os.system(cmd)