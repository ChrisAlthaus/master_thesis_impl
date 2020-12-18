import random
import numpy as np
import datetime
import json
import os
import time
import csv

inputfile = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/out/predictions/graphs/12-10_17-57-13/.descriptors/graphdescriptors.json'
    #'/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/out/predictions/graphs/12-10_17-57-13/.descriptors/graphdescriptors.json' #/nfs/data/iart/kaggle/img
    #'/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/out/predictions/graphs/12-09_17-22-49/.descriptors/graphdescriptors.json' #/home/althausc/nfs/data/coco_17/test_original10k
    #'/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/out/predictions/graphs/12-08_18-42-37/.descriptors/graphdescriptors.json'

epochs = 40
traineval_epoch = 5
valeval_epoch = 5
saveepoch = 10
valsize = 0.2
evaltopk = 100

dimensionsize = 132  #32, 64, 128, 132, 256, 512, 1024, 2048
wliters = 3
downsampling = 0.0001 #default: 0.0001
lr = 0.025 #0.025 #default: 0.025
stepsinfer = 100 #20 #50 #100
#Without upsampling the median feature dimension was: 132. (/home/althausc/master_thesis_impl/graph2vec/models/12-12_10-34-58-kaggle-from-here-on/config.txt)
minfeaturedim = 132

gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/sbatch_nogpu.sh'
jobname = 'graph2vec-train%s'%datetime.datetime.now().strftime('%d-%H-%M-%S')

logdir = '/home/althausc/master_thesis_impl/graph2vec/models/logs'
logfile = os.path.join(logdir, '%s.log'%datetime.datetime.now().strftime('%m-%d-%H-%M-%S'))

cmd =  ("sbatch -w devbox4 -J {} -o {} "+ \
            "{} python3.6 -u /home/althausc/master_thesis_impl/graph2vec/src/graph2vec.py "+ \
            "--input-path {} --output-path {} --workers 4 --dimensions {} --epochs {} --wl-iterations {} --down-sampling {} --learning-rate {} --steps-inference {} --min-featuredim {} "+ \
			"--epochsave {} --traineval {} --valeval {} --valsize {} --evaltopk {}")\
    			.format(jobname, logfile, gpu_cmd, inputfile, 'notused', dimensionsize, epochs, wliters, downsampling, lr, stepsinfer, minfeaturedim,
                        saveepoch, traineval_epoch, valeval_epoch, valsize, evaltopk)
print(cmd)
os.system(cmd)