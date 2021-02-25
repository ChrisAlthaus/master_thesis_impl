import os
import random
import time

gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun1-2.sh'#'/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun1-2-qrtx8000.sh' #'/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun1-2.sh'#'/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_run-2.sh' #
jobname = 'sg-eval'
masterport = random.randint(10020, 10100)
_NUMGPUS = 1
configfile ='/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/configs/e2e_relation_X_101_32_8_FPN_1x.yaml'
glovedir = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/sgdet_training/glove'
cpktdir = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/sgdet_training/causal_motif_sgdet' 
    #'/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/sgdet_training/12-02_09-23-52-dev3'  #
    #'/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/sgdet_training/12-10_08-49-49-dev3' 
    # #'/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/sgdet_training/causal_motif_sgdet'
effecttype = 'none'#'TDE' #'NIE', 'TE', 'none'
fusiontype = 'sum' #'gate'
outputdir = os.path.join(cpktdir, '.eval-aps')
if not os.path.isdir(outputdir):
    os.makedirs(outputdir)
logfile = os.path.join(outputdir, 'evallog-{}.txt'.format(effecttype))

evalscript = '/home/althausc/master_thesis_impl/scripts/scenegraph/utils/evalmodel.py' #'/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/tools/relation_test_net_original.py'

 #Print console cmd for better debugging
cmd = ("sbatch -w devbox4 -J {} -o {} "+ \
"{} python3.6 -m torch.distributed.launch --master_port {} --nproc_per_node={} "+\
"{} \t"+\
"--config-file \"{}\" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.RELATION_ON True "+\
"MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE {} MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE {} "+\
"MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs  TEST.IMS_PER_BATCH 1 DTYPE \"float16\" "+ \
"GLOVE_DIR {} MODEL.PRETRAINED_DETECTOR_CKPT {} OUTPUT_DIR {}")\
	.format(jobname, logfile, gpu_cmd, masterport, _NUMGPUS, evalscript, configfile, effecttype, fusiontype, glovedir, cpktdir, outputdir)
    
print(cmd)
os.system(cmd)
time.sleep(10)

