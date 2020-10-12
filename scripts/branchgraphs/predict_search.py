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
parser.add_argument('-inputImg',required=True,
                    help='Image for which to infer best k matching images.')
args = parser.parse_args()

#Folder with input image only
img_dir = os.path.dirname(args.inputImg)

_PRINT_CMDS = True
_EXEC_CMDS = False

def latestdir(dir):
    diritems = [os.path.join(dir, d) for d in os.listdir(dir)]
    all_subdirs = [d for d in diritems if os.path.isdir(d)]
    return max(all_subdirs, key=os.path.getmtime)

def filewithname(dir, searchstr):
    for item in os.listdir(dir):
        if os.path.isfile(os.path.join(dir,item)) and searchstr in item:
            return os.path.join(dir,item)
    return None


# ----------------- SCENE GRAPH PREDICTION ---------------------
print("SCENE GRAPH PREDICTION:")
gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun_G1d4-2.sh'
model_dir = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/causal_motif_sgdet'
out_dir = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/out/predictions/single'

_EFFECT_TYPES = ['none', 'TDE', 'NIE', 'TE']
_FUSION_TYPES = ['sum', 'gate']
_CONTEXTLAYER_TYPES = ['motifs', 'vctree', 'vtranse']

effect_type = _EFFECT_TYPES[1]
fusion_type = _FUSION_TYPES[0]
contextlayer_type = _CONTEXTLAYER_TYPES[0] 

#filter prediction settings
topkboxes = 20
topkrels = 40
ftresh_boxes = 0.5
ftresh_rels = 0.5

cmd = ("{} python3.6 -m torch.distributed.launch --master_port 10027 --nproc_per_node=1 \t"+ \
	 			"/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/tools/relation_test_net.py \t" +\
	            "--config-file \"/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/configs/e2e_relation_X_101_32_8_FPN_1x.yaml\"  \t" +\
	            "MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \t" +\
	            "MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \t" +\
	            "MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor \t" +\
	            "MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE {} \t" +\
	            "MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE {} \t" +\
	            "MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER {} \t" +\
	            "TEST.IMS_PER_BATCH 1 \t" +\
	            "DTYPE \"float16\" \t" +\
	            "GLOVE_DIR /home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/glove \t" +\
	            "MODEL.PRETRAINED_DETECTOR_CKPT {} \t" +\
                "SOLVER.PRE_VAL False \t" +\
	            "OUTPUT_DIR {} \t" +\
	            "TEST.CUSTUM_EVAL True \t" +\
	            "TEST.CUSTUM_PATH {} \t" +\
	            "DETECTED_SGG_DIR {} \t" +\
                "topkboxes {} topkrels {} \t"+\
				"filtertresh_boxes {} filtertresh_rels {} \t").format(gpu_cmd, effect_type, fusion_type, contextlayer_type, model_dir, model_dir, img_dir, out_dir,
                                                                        topkboxes, topkrels, ftresh_boxes, ftresh_rels)
if _PRINT_CMDS:
    print(cmd)
if _EXEC_CMDS:
    os.system(cmd)

outrun_dir = latestdir(out_dir)
print("Output Directory: %s\n"%out_dir)


# ----------------------- VISUALIZE SCENE GRAPH -----------------------
print("VISUALIZE SCENEGRAPH ...")
out_dir = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/out/visualize'
filterlabels = True
cmd = "python3.6 /home/althausc/master_thesis_impl/scripts/scenegraph/visualizeimgs.py -predictdir {} {}"\
														.format(outrun_dir, '-filter' if filterlabels else ' ')
if _PRINT_CMDS:
    print(cmd)
if _EXEC_CMDS:
    os.system(cmd)

#outrun_dir = latestdir(out_dir)
print("Output Directory: %s\n"%out_dir)


# ----------------- TRANSFORM PREDICTIONS INTO GRAPH2VEC FORMAT ---------------
print("TRANSFORM PREDICTIONS INTO GRAPH2VEC FORMAT")
out_dir = "/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/out/topk/single"
pred_imginfo = os.path.join(outrun_dir, 'custom_data_info.json')
pred_file = os.path.join(outrun_dir, 'custom_prediction.json')
relasnodes = True

cmd = "python3.6 /home/althausc/master_thesis_impl/scripts/scenegraph/filter_resultgraphs.py -file {} -imginfo {} -outputdir {} -build_labelvectors {}" \
			 .format(pred_file, pred_imginfo, out_dir, '-relsasnodes' if relasnodes else ' ')				
if _PRINT_CMDS:
    print(cmd)
if _EXEC_CMDS:
    os.system(cmd)

outrun_dir = latestdir(out_dir)
print("Output Directory: %s\n"%out_dir)


# ----------------- GRAPH2VEC PREDICTION & RETRIEVAL ---------------------
print("GRAPH2VEC PREDICTION & RETRIEVAL ...")
modeldir = latestdir('/home/althausc/master_thesis_impl/graph2vec/models') #'/home/althausc/master_thesis_impl/graph2vec/models/09/22_09-58-49'
g2v_model = os.path.join(modeldir, filewithname(modeldir, 'g2vmodel')) 
labelvecpath = os.path.join(modeldir, 'labelvectors-topk.json')
inputfile = os.path.join(outrun_dir, 'graphs-topk.json')
_REWEIGHT_MODES = ['jaccard', 'euclid']

topk = 10
reweight = True
r_mode = _REWEIGHT_MODES[0]

if reweight:
    cmd = ("python3.6 /home/althausc/master_thesis_impl/retrieval/graph_search.py " +\
                    "--model {} --inputpath {} " +\
				    "--inference --topk {} " +\
                    "--reweight --reweightmode {} " +\
                    "--labelvecpath {}").format(g2v_model, inputfile, topk, r_mode, labelvecpath)         
else:
    cmd = ("python3.6 /home/althausc/master_thesis_impl/retrieval/graph_search.py" +\
									"--model {} --inputpath {} --inference --topk {}")\
										.format(g2v_model, inputfile, topk)
if _PRINT_CMDS:
    print(cmd)
if _EXEC_CMDS:
    os.system(cmd)
        
out_dir = '/home/althausc/master_thesis_impl/retrieval/out/scenegraphs'
outrun_dir = latestdir(out_dir)
print("Output Directory: %s\n"%out_dir)

