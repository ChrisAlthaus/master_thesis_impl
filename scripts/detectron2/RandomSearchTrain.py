import random
import numpy as np
import datetime
import json
import os
import time

#Perform random search over the hyper-parameters
_PARAM_MODES = ['originalpaper', 'detectrondefault', 'randomsearch', 'custom']
_PARAM_MODE = _PARAM_MODES[3]
_NUM_RUNS = 1

_TRAINMODES = ["ALL", "RESNETF", "RESNETL", "HEADSALL", 'SCRATCH']
_DATA_AUGM = [True, False]
_LRS = [0.01, 0.001, 0.0001, 0.00001]
_BN = [True, False]
_MINKPTS = [1,2]
_NUMEPOCHS = 10
_STEPS_GAMMA = [ [[0.76, 0.92], 0.1], [np.linspace(0.7, 1, 10).tolist(), 0.5] ]
_MINSCALES = [(640, 672, 704, 736, 768, 800), [512], [800]]
_IMSPERBATCH = [2, 4]
_NUMGPUS = 1
_RPN_POSITIVE_RATIOS = [0.33, 0.5]
_GRADIENT_CLIP_VALUE = [1, 5]

for i in range(0,_NUM_RUNS):

    if _PARAM_MODE == 'originalpaper': #see https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/config.py or https://arxiv.org/pdf/1703.06870.pdf
        trainmode = 'ALL'
        dataaugm = False
        lr = 0.001 #0.02/(16/_IMSPERBATCH) #original trained with batchsize=16
        bn = False
        minscales = (640, 672, 704, 736, 768, 800)
        rpn_posratio = 0.33
        gradient_clipvalue = 1 # actually:5, but expl grads
        gamma = 0.1
        steps = [0.76, 0.92]
        minkpts = 1 #? not stated

    elif _PARAM_MODE == 'detectrondefault':
        trainmode = 'ALL'
        dataaugm = False
        batchsize = 4
        lr = 0.02/(16/_IMSPERBATCH) #original trained with batchsize=16
        bn = False
        minscales = (640, 672, 704, 736, 768, 800)
        rpn_posratio = 0.5
        gradient_clipvalue = 1  
        minkpts = 1
        steps = [0.76, 0.92] 
        gamma = 0.1

    elif _PARAM_MODE == 'randomsearch':
        trainmode = 'ALL'
        dataaugm = random.choice(_DATA_AUGM)
        batchsize = random.choice(_IMSPERBATCH)
        lr = 0.001 #random.choice(_LRS)
        bn = random.choice(_BN)
        minkpts = 1 #choice?
        step_gam = random.choice(_STEPS_GAMMA)
        steps = step_gam[0]
        gamma = step_gam[1]
        minscales = random.choice(_MINSCALES)
        rpn_posratio = 0.5
        gradient_clipvalue = 1 

    elif _PARAM_MODE == 'custom':
        trainmode = 'ALL'
        dataaugm = True
        batchsize = 4
        lr = 0.001
        bn = True
        minkpts = 1 
        steps = np.linspace(0.7, 1, 10).tolist()
        gamma = 0.75  #0.75 ^ 10 = 0.05
        minscales = (512,)
        rpn_posratio = 0.5
        gradient_clipvalue = 1

    else:
        raise ValueError()



    print(" --------------------------- RUN %d ------------------------------ "%i)
    params = {'trainmode': trainmode, 'dataaugm': dataaugm, 'batchsize': batchsize, 'epochs': _NUMEPOCHS, 'lr': lr, 'bn': bn, 'minscales': minscales, 'rpn_posratio': rpn_posratio,
            'gradient_clipvalue': gradient_clipvalue, 'steps': steps, 'gamma': gamma, 'minkpt': minkpts}
    print("Parameters: ",params)
    logdir = os.path.join('/home/althausc/master_thesis_impl/detectron2/out/checkpoints/trainconfigs_tmp', datetime.datetime.now().strftime('%m-%d_%H-%M-%S')+ '_%d'%i)
    os.makedirs(logdir)
    paramconfig = os.path.join(logdir, 'paramsconfig.txt')

    with open(paramconfig, 'w') as f:
        json.dump(params, f)

    #Normal run command, Print for eventually debugging
    gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun_G1d4-1.sh'
    resume = False #True
    maskrcnn_cp = '/home/althausc/master_thesis_impl/detectron2/out/checkpoints/08/07_12-40-41_all/model_0214999.pth' #

    #Additional params to setup in file: folder : BN[yes,no], LR, Weight Decay[steps& exp/no exp], Data Augmentation[Random Rotation, Flip & Crop],
    #                                             Min keypoints[for filter images], Additional[MinSizeTrain, ImgPerBatch]

    cmd = "{} python3.6 /home/althausc/master_thesis_impl/scripts/detectron2/MaskRCNN_train_styletransfer.py -paramsconfig {} -addconfig"\
                                                    .format(gpu_cmd, paramconfig, '-resume %s'%maskrcnn_cp if resume else ' ')

    print(cmd)

    #Start sbatch training
    gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun1-1.sh'
    resume = False #True
    maskrcnn_cp = '/home/althausc/master_thesis_impl/detectron2/out/checkpoints/08/07_12-40-41_all/model_0214999.pth'
    jobname = "maskrcnn-train-%s"%datetime.datetime.now().strftime('%d_%H-%M-%S')
    logfile = os.path.join(logdir, 'train.log')

    #Additional params to setup in file: folder : BN[yes,no], LR, Weight Decay[steps& exp/no exp], Data Augmentation[Random Rotation, Flip & Crop],
    #                                             Min keypoints[for filter images], Additional[MinSizeTrain, ImgPerBatch]

    cmd = ("sbatch -w devbox4 -J {} -o {} "+ \
        "{} python3.6 /home/althausc/master_thesis_impl/scripts/detectron2/MaskRCNN_train_styletransfer.py -paramsconfig {} -addconfig")\
                                                    .format(jobname, logfile, gpu_cmd, paramconfig, '-resume %s'%maskrcnn_cp if resume else ' ')

    print(cmd)
    os.system(cmd)
    time.sleep(10)
    print()