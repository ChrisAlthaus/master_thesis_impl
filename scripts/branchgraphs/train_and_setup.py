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
import random

def latestdir(dir):
    diritems = [os.path.join(dir, d) for d in os.listdir(dir)]
    all_subdirs = [d for d in diritems if os.path.isdir(d)]
    return max(all_subdirs, key=os.path.getmtime)

def filewithname(dir, searchstr):
    for item in os.listdir(dir):
        if os.path.isfile(os.path.join(dir,item)) and searchstr in item:
            return os.path.join(dir,item)
    return None

_PRINT_CMDS = True
_EXEC_CMDS = False


# ---------------- PREPARE SCENE GRAPH DATA --------------------
print("PREPARE SCENE GRAPH DATA:")
#Update image info because of width & height
imginfo = '/home/althausc/nfs/data/vg/image_data.json'
imgdir = '/home/althausc/nfs/data/vg_styletransfer/VG_100K'
out_dir = '/home/althausc/nfs/data/vg_styletransfer'

cmd = "python3.6 /home/althausc/master_thesis_impl/scripts/scenegraph/update_imageinfo.py -file {} -imagedir {} -outputdir {}"\
									.format(imginfo, imgdir, out_dir)
if _PRINT_CMDS:
    print(cmd)
if _EXEC_CMDS:
    os.system(cmd)

#Update scene graph respectively filter out not wanted bbox & rel classes
sgraphfile = '/home/althausc/nfs/data/vg/VG-SGG-with-attri.h5'
out_dir = '/home/althausc/nfs/data/vg_styletransfer'
trainandtest = True #whether to transform both train & validation split or just validation split

cmd = "python3.6 /home/althausc/master_thesis_impl/scripts/scenegraph/update_scenegraphs.py -file {} -outputdir {} {}"\
																.format(sgraphfile, out_dir, '-trainandtest' if trainandtest else '')
if _PRINT_CMDS:
    print(cmd)
if _EXEC_CMDS:
    os.system(cmd)

print("Output Directory: %s\n"%out_dir)


# ------------------- SCENE GRAPH FASTER-RCNN TRAINING -----------------------
print("SCENE GRAPH FASTER-RCNN TRAINING:")
gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun_G4d4-2.sh'
out_dir = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/faster_rcnn_trained'

cmd = ("{} python3.6 -m torch.distributed.launch --master_port 27000 --nproc_per_node=4" +\
	"/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/tools/detector_pretrain_net.py "+\
	"--config-file \"configs/e2e_relation_detector_X_101_32_8_FPN_1x.yaml\" \t"+\
	"SOLVER.IMS_PER_BATCH 8 \t"+\
	"TEST.IMS_PER_BATCH 4 \t"+\
	"DTYPE \"float16\" \t"+\
	"SOLVER.MAX_ITER 50000 \t"+\
	"SOLVER.STEPS \"(30000, 45000)\" \t"+\
	"SOLVER.VAL_PERIOD 2000 \t"+\
	"SOLVER.CHECKPOINT_PERIOD 2000 \t"+\
	"MODEL.RELATION_ON False \t"+\
	"OUTPUT_DIR {} \t"+\
	"SOLVER.PRE_VAL False \t"+\
	"MODEL.RPN.ROI_BOX_HEAD.NUM_CLASSES 65")\
		.format(gpu_cmd, out_dir)
if _PRINT_CMDS:
    print(cmd)
if _EXEC_CMDS:
    os.system(cmd)

print("Output Directory: %s\n"%out_dir)


# ------------------- SCENE GRAPH GENERATION TRAINING -----------------------
#Also see for an updated version: master_thesis_impl/scripts/scenegraph/RandomSearchTrain.py 
print("SCENE GRAPH FASTER-RCNN TRAINING:")
gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun_G2d4-2.sh'
pretrained_frcnn = 'master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/pretrained_faster_rcnn/model_final.pth' # os.path.join(out_dir, 'model_final.pth') #or /home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/pretrained_faster_rcnn

#notes: effect type only used for prediction/inference
_PREDICTOR = ['MotifPredictor', 'IMPPredictor', 'VCTreePredictor', 'TransformerPredictor', 'CausalAnalysisPredictor']
_FUSION_TYPES = ['sum', 'gate']
_CONTEXTLAYER_TYPES = ['motifs', 'vctree', 'vtranse']

predictor = _PREDICTOR[4]
fusion_type = _FUSION_TYPES[0]
contextlayer_type = _CONTEXTLAYER_TYPES[0]
#out_dir = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/%s_%s_%s_sgdet'%(predictor, contextlayer_type, fusion_type)
out_dir = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints'

#sgdet: parameter MODEL.ROI_RELATION_HEAD.REQUIRE_BOX_OVERLAP = True
cmd = ("{} python3.6 -m torch.distributed.launch --master_port 10026 --nproc_per_node=2 "+\
	"/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/tools/relation_train_net.py \t"+\
	"--config-file \"/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/configs/e2e_relation_X_101_32_8_FPN_1x.yaml\" \t"+\
	"MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \t"+\
	"MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \t"+\
	"MODEL.ROI_RELATION_HEAD.PREDICTOR {} \t"+\
	"MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE none \t"+\
	"MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE {} \t"+\
	"MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER {} \t"+\
	"MODEL.ROI_RELATION_HEAD.REQUIRE_BOX_OVERLAP True \t"+\
	"SOLVER.IMS_PER_BATCH 12 \t"+\
	"TEST.IMS_PER_BATCH 2 \t"+\
	"DTYPE \"float16\" \t"+\
	"SOLVER.MAX_ITER 50000 \t"+\
	"SOLVER.VAL_PERIOD 2000 \t"+\
	"SOLVER.CHECKPOINT_PERIOD 2000 \t"+\
	"GLOVE_DIR {} \t"+\
	"MODEL.PRETRAINED_DETECTOR_CKPT {} \t"+\
	"OUTPUT_DIR {}")\
		.format(gpu_cmd, predictor, fusion_type, contextlayer_type, out_dir, pretrained_frcnn, out_dir)
if _PRINT_CMDS:
    print(cmd)
if _EXEC_CMDS:
    os.system(cmd)

print("Output Directory: %s\n"%out_dir)

"""
sbatch -w devbox4 -J graphprediction -o /home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/out/predictions/graphs/logs/02-08_21-39-04.txt /home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun1-2.sh python3.6 -m torch.distributed.launch     --master_port 10056     --nproc_per_node=1 /home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/tools/relation_test_net.py    --config-file "/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/configs/e2e_relation_X_101_32_8_FPN_1x.yaml"     MODEL.ROI_RELATION_HEAD.USE_GT_BOX False        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False        MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor       MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE  MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum  MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs     TEST.IMS_PER_BATCH 1     DTYPE "float16"         GLOVE_DIR /home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/sgdet_training/glove      MODEL.PRETRAINED_DETECTOR_CKPT /home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/sgdet_training/12-02_09-23-52-dev3   OUTPUT_DIR /home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/sgdet_training/12-02_09-23-52-dev3        TEST.CUSTUM_EVAL True   TEST.CUSTUM_PATH /home/althausc/nfs/data/artimages/painterbynumbers10k       TEST.POSTPROCESSING.TOPKBOXES -1        TEST.POSTPROCESSING.TOPKRELS 75  TEST.POSTPROCESSING.TRESHBOXES 0.17     TEST.POSTPROCESSING.TRESHRELS 0.15       DETECTED_SGG_DIR /home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/out/predictions/graphs
"""

# ----------------- SCENE GRAPH PREDICTION ---------------------
print("SCENE GRAPH PREDICTION:")
gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun1-2.sh'
#Note: model directory should contain a file 'last_checkpoint' with path to the used checkpoint
model_dir = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/sgdet_training/12-02_09-23-52-dev3'
            #'/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/others/causal_motif_sgdet'           
out_dir = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/out/predictions/graphs'

_EFFECT_TYPES = ['none', 'TDE', 'NIE', 'TE']
_FUSION_TYPES = ['sum', 'gate']
_CONTEXTLAYER_TYPES = ['motifs', 'vctree', 'vtranse']

effect_type = _EFFECT_TYPES[1]
fusion_type = _FUSION_TYPES[0]
contextlayer_type = _CONTEXTLAYER_TYPES[0] 

topkboxes = -1 #4#10
topkrels = 75#10#20
#fine-tuned by visual inspection 
# -> abstract art predictions should be partially prevented (/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/out/predictions/graphs/12-10_09-47-30/.visimages/5878_1scenegraph.jpg)
treshboxes = 0.12 
treshrels = 0.1

masterport = random.randint(10020, 10100)

jobname = 'graphprediction'
logfile = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/out/predictions/graphs/logs/%s.txt'%datetime.datetime.now().strftime('%m-%d_%H-%M-%S')

#Note: - MODEL.PRETRAINED_DETECTOR_CKPT same functionality as OUTPUT_DIR (but OUTPUT_DIR used for model loading)
#      - TEST.IMS_PER_BATCH has to be 1 (assertion)
cmd = ("sbatch -w devbox4 -J {} -o {} "+ \
				"{} python3.6 -m torch.distributed.launch" +\
                "\t --master_port {}" +\
                "\t --nproc_per_node=1 /home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/tools/relation_test_net.py" +\
                "\t --config-file \"/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/configs/e2e_relation_X_101_32_8_FPN_1x.yaml\" " +\
                "\t MODEL.ROI_RELATION_HEAD.USE_GT_BOX False" +\
                "\t MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False" +\
                "\t MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor" +\
                "\t MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE {}" +\
                "\t MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE {}" +\
                "\t MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER {}" +\
                "\t TEST.IMS_PER_BATCH 1" +\
                "\t DTYPE \"float16\"" +\
                "\t GLOVE_DIR /home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/sgdet_training/glove" +\
                "\t MODEL.PRETRAINED_DETECTOR_CKPT {}" +\
                "\t OUTPUT_DIR {}" +\
                "\t TEST.CUSTUM_EVAL True" +\
                "\t TEST.CUSTUM_PATH {}" +\
                "\t TEST.POSTPROCESSING.TOPKBOXES {}" +\
                "\t TEST.POSTPROCESSING.TOPKRELS {}" +\
                "\t TEST.POSTPROCESSING.TRESHBOXES {}" +\
                "\t TEST.POSTPROCESSING.TRESHRELS {}" +\
                "\t DETECTED_SGG_DIR {} \t").format(jobname, logfile, gpu_cmd, masterport, effect_type, fusion_type, contextlayer_type, model_dir, model_dir, imgdir,
                                                          topkboxes, topkrels, treshboxes, treshrels, out_dir)
if _PRINT_CMDS:
    print(cmd)
if _EXEC_CMDS:
    os.system(cmd)

print("Output Directory: %s\n"%out_dir)
outrun_dir = latestdir(out_dir)

# ------------------------------ VISUALIZE PREDICTIONS -------------------------------
print("VISUALIZE SCENEGRAPH ...")

gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/sbatch_nogpu.sh'
jobname = 'graphvisualization'

predictdir = outrun_dir #'/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/out/predictions/graphs/12-07_14-09-00' #e.g.
logfile = os.path.join(predictdir, 'vislog.txt')

cmd = ("sbatch -w devbox4 -J {} -o {}"+ \
             "{} python3.6 -u /home/althausc/master_thesis_impl/scripts/scenegraph/visualizeimgs.py -predictdir {} -visrandom")\
                            .format(jobname, logfile, gpu_cmd, predictdir)
if _PRINT_CMDS:
    print(cmd)
if _EXEC_CMDS:
    os.system(cmd)

print("VISUALIZE SCENEGRAPH DONE.\n")


# ----------------- TRANSFORM PREDICTIONS INTO GRAPH2VEC FORMAT ---------------
print("TRANSFORM PREDICTIONS INTO GRAPH2VEC FORMAT")

pred_imginfo = os.path.join(outrun_dir, 'custom_data_info.json')
pred_file = os.path.join(outrun_dir, 'custom_prediction.json')
relasnodes = True

cmd = ("python3.6 /home/althausc/master_thesis_impl/scripts/graph_descriptors/graphdescriptors.py "+ \
	         		"-file {} -imginfo {} -build_labelvectors {} ")\
					.format(pred_file, pred_imginfo, '-relsasnodes' if relasnodes else ' ')
if _PRINT_CMDS:
    print(cmd)
if _EXEC_CMDS:
    os.system(cmd)

print("Output Directory: %s\n"%out_dir)
outrun_dir = latestdir(out_dir)
# default for testing: outrun_dir = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/out/topk/graphs/09-28_13-15-09'


# ----------------- GRAPH2VEC TRAINING ---------------------
print("GRAPH2VEC TRAINING:")

inputfile = os.path.join(outrun_dir, 'graphs-topk.json')
model_dir = '/home/althausc/master_thesis_impl/graph2vec/models'
valepoch = 5
valsize = 0.2
evaltopk = 100
saveepoch = 20

cmd = ("python3.6 /home/althausc/master_thesis_impl/graph2vec/src/graph2vec.py "+ \
            "--input-path {} --output-path {} --workers 4 --dimensions 128 --epochs 1 --wl-iterations 2 --down-sampling 0.0001 "+ \
			"--epochsave {} --valeval {} --valsize {} --evaltopk {}")\
				.format(inputfile, 'notused', saveepoch, valepoch, valsize, evaltopk)
if _PRINT_CMDS:
    print(cmd)
if _EXEC_CMDS:
    os.system(cmd)

print("Output Directory: %s\n"%model_dir)
outrun_dir = latestdir(model_dir)


