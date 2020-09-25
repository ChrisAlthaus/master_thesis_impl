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
parser.add_argument('-imagedir',required=True,
                    help='Image directory for which to predict scene graphs & build g2v model.')
args = parser.parse_args()

logpath = '/home/althausc/master_thesis_impl/scripts/branchgraphs/trainlogs'
if not os.path.exists(logpath):
    os.makedirs(logpath)
    print("Successfully created output directory: ", logpath)

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
gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun_G1d4.sh'
model_dir = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/causal_motif_sgdet'
out_dir = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/out/predictions/graphs'

_EFFECT_TYPES = ['none', 'TDE', 'NIE', 'TE']
_FUSION_TYPES = ['sum', 'gate']
_CONTEXTLAYER_TYPES = ['motifs', 'vctree', 'vtranse']

effect_type = _EFFECT_TYPES[1]
fusion_type = _FUSION_TYPES[0]
contextlayer_type = _CONTEXTLAYER_TYPES[0] 
logfile = os.path.join(logpath, '1-prediction.txt')

print("{} python3.6 -m torch.distributed.launch \
	            --master_port 10027 \
	            --nproc_per_node=1 /home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/tools/relation_test_net.py \
	            --config-file \"configs/e2e_relation_X_101_32_8_FPN_1x.yaml\"  \
	            MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
	            MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
	            MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor \
	            MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE {} \
	            MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE {} \
	            MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER {} \
	            TEST.IMS_PER_BATCH 1 \
	            DTYPE \"float16\" \
	            GLOVE_DIR /home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/glove \
	            MODEL.PRETRAINED_DETECTOR_CKPT {} \
	            OUTPUT_DIR {} \
	            TEST.CUSTUM_EVAL True \
	            TEST.CUSTUM_PATH {} \
	            DETECTED_SGG_DIR {} &> {}".format(gpu_cmd, effect_type, fusion_type, contextlayer_type, model_dir, model_dir, args.imagedir, out_dir, logfile))

os.chdir('/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch')
if os.system("{} python3.6 -m torch.distributed.launch \
	            --master_port 10027 \
	            --nproc_per_node=1 tools/relation_test_net.py \
	            --config-file \"configs/e2e_relation_X_101_32_8_FPN_1x.yaml\"  \
	            MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
	            MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
	            MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor \
	            MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE {} \
	            MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE {} \
	            MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER {} \
	            TEST.IMS_PER_BATCH 1 \
	            DTYPE \"float16\" \
	            GLOVE_DIR /home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/glove \
	            MODEL.PRETRAINED_DETECTOR_CKPT {} \
	            OUTPUT_DIR {} \
	            TEST.CUSTUM_EVAL True \
	            TEST.CUSTUM_PATH {} \
	            DETECTED_SGG_DIR {} &> {}".format(gpu_cmd, effect_type, fusion_type, contextlayer_type, model_dir, model_dir, args.imagedir, out_dir, logfile)):
	raise RuntimeError('Scene graph training failed.') 

outrun_dir = latestdir(out_dir)
print("\n\n")

# ----------------- TRANSFORM PREDICTIONS INTO GRAPH2VEC FORMAT ---------------
print("TRANSFORM PREDICTIONS INTO GRAPH2VEC FORMAT")

pred_imginfo = os.path.join(outrun_dir, 'custom_data_info.json')
pred_file = os.path.join(outrun_dir, 'custom_prediction.json')
relasnodes = True
out_dir = "/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/out/topk/graphs"
logfile = os.path.join(logpath, '2-transformg2vformat.txt')

print("python3.6 filter_resultgraphs.py \
				-file {} \
				-imginfo {} \
				-outputdir {} \
				-build_labelvectors \
				{} &> {}".format(pred_file, pred_imginfo, out_dir, '-relasnodes' if relasnodes else ' ', logfile))

os.chdir('/home/althausc/master_thesis_impl/scripts/scenegraph')
if os.system("python3.6 filter_resultgraphs.py \
				-file {} \
				-imginfo {} \
				-outputdir {} \
				-build_labelvectors \
				{} &> {}".format(pred_file, pred_imginfo, out_dir, '-relasnodes' if relasnodes else ' ', logfile)):
	raise RuntimeError('Transform into G2V format failed.') 

outrun_dir = latestdir(out_dir)
print("\n\n")

# ----------------- GRAPH2VEC TRAINING ---------------------
print("GRAPH2VEC TRAINING:")

inputfile = os.path.join(outrun_dir, 'graphs-topk.json')
model_dir = '/home/althausc/master_thesis_impl/graph2vec/models'
logfile = os.path.join(logpath, '3-g2vtrain.txt')

print("python3.6 /home/althausc/master_thesis_impl/graph2vec/src/graph2vec.py \
            --input-path {} \
            --output-path {} \
            --workers 4 \
            --dimensions 128 \
            --epochs 1 \
            --wl-iterations 2 \
            --down-sampling 0.0001 &> {}".format(inputfile, 'not used', logfile))

os.chdir('/home/althausc/master_thesis_impl/graph2vec')
if os.system("python3.6 src/graph2vec.py \
            --input-path {} \
            --output-path {} \
            --workers 4 \
            --dimensions 128 \
            --epochs 1 \
            --wl-iterations 2 \
            --down-sampling 0.0001 &> {}".format(inputfile, 'not used', logfile)):
	raise RuntimeError('G2V training failed.') 

outrun_dir = latestdir(model_dir)
print("Graph2Vec model path: ",outrun_dir)

