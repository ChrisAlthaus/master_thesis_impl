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
import shutil


"""parser = argparse.ArgumentParser()
parser.add_argument('-inputImg',required=True,
                    help='Image for which to infer best k matching images.')
args = parser.parse_args()

#Folder with input image only
img_dir = os.path.dirname(args.inputImg)"""

logpath = '/home/althausc/master_thesis_impl/results/logs/jupyter-notebook/graphbranch'
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

def predict_scenegraph(imagepath):
    #Create a tmp image dir
    img_dir = os.path.join('/home/althausc/master_thesis_impl/scripts/branchgraphs/images', datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    else:
        raise ValueError("Output directory %s already exists."%img_dir)
    print(imagepath)
    shutil.copyfile(imagepath, os.path.join(img_dir, os.path.basename(imagepath)))

    # ----------------- SCENE GRAPH PREDICTION ---------------------
    print("SCENE GRAPH PREDICTION:")
    gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun_G1d4.sh'
    model_dir = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/causal_motif_sgdet'
    out_dir = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/out/predictions/graphs'
    logfile = os.path.join(logpath, '1-scenegraph.txt')

    _EFFECT_TYPES = ['none', 'TDE', 'NIE', 'TE']
    _FUSION_TYPES = ['sum', 'gate']
    _CONTEXTLAYER_TYPES = ['motifs', 'vctree', 'vtranse']

    effect_type = _EFFECT_TYPES[1]
    fusion_type = _FUSION_TYPES[0]
    contextlayer_type = _CONTEXTLAYER_TYPES[0] 

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
                    DETECTED_SGG_DIR {} &> {}".format(gpu_cmd, effect_type, fusion_type, contextlayer_type, model_dir, model_dir, img_dir, out_dir, logfile)):
        raise RuntimeError('Scene graph prediction failed.')

    out_dir = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/out/predictions/single'
    outrun_dir = latestdir(out_dir)
    print('')
    return outrun_dir

def visualize_scenegraph(anndir):
    logfile = os.path.join(logpath, '2-visualize.txt')

    os.chdir("/home/althausc/master_thesis_impl/scripts/scenegraph")
    if os.system("python3.6 visualizeimgs.py -predictdir {} &> {}".format(anndir, logfile)):
        raise RuntimeError('Scene graph visualization failed.')

    outrun_dir = latestdir('/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/out/visualize')
    img_files = [os.path.join(outrun_dir, f) for f in os.listdir(outrun_dir) if os.path.isfile(os.path.join(outrun_dir, f))]
    return img_files

def transform_into_g2vformat(anndir):
    # ----------------- TRANSFORM PREDICTIONS INTO GRAPH2VEC FORMAT ---------------
    print("TRANSFORM PREDICTIONS INTO GRAPH2VEC FORMAT")

    pred_imginfo = os.path.join(anndir, 'custom_data_info.json')
    pred_file = os.path.join(anndir, 'custom_prediction.json')
    out_dir = "/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/out/topk/single"

    os.chdir('/home/althausc/master_thesis_impl/scripts/scenegraph')
    os.system("python3.6 filter_resultgraphs.py \
                    -file {} \
                    -imginfo {} \
                    -outputdir {}".format(pred_file, pred_imginfo, out_dir))

    outrun_dir = latestdir(out_dir)
    graphfile = os.path.join(outrun_dir, 'graphs-topk.json')
    print('')
    return graphfile

def search_topk(graphfile, k):
    # ----------------- GRAPH2VEC PREDICTION & RETRIEVAL ---------------------
    print("GRAPH2VEC PREDICTION & RETRIEVAL:")
    g2v_model = '/home/althausc/master_thesis_impl/graph2vec/models/09/22_09-58-49/g2vmodel'
    inputfile = graphfile
    topk = k

    os.system("python3.6 /home/althausc/master_thesis_impl/retrieval/graph_search.py --model {} --input-path {} \
    				 --inference --topk {}".format(g2v_model, inputfile, topk))

    out_dir = '/home/althausc/master_thesis_impl/retrieval/out/scenegraphs/09'
    outrun_dir = latestdir(out_dir)
    with open(os.path.join(outrun_dir,"topkresults.json"), 'r') as f:
        json_data = json.load(f)

    return json_data


