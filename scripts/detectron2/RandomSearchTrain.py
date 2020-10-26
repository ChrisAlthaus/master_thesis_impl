import random
import numpy as np
import datetime
import json
import os
import time
import csv
import argparse

#Perform random search over the hyper-parameters
_PARAM_MODES = ['originalpaper', 'detectrondefault', 'randomsearch', 'custom', 'loadparams']
_PARAM_MODE = _PARAM_MODES[3]
_NUM_RUNS = 1

_TRAINMODES = ["ALL", "RESNETF", "RESNETL", "HEADSALL", 'SCRATCH']
_DATA_AUGM = [True, False]
_LRS = [0.01, 0.001, 0.0001, 0.00001]
_BN = [True, False]
_MINKPTS = [1,2,4]
_NUMEPOCHS = 10
_STEPS_GAMMA = [ [[0.76, 0.92], 0.1], [np.linspace(0.7, 0.92, 6).tolist(), 0.5] ]#[np.linspace(0.7, 1, 4).tolist(), 0.5]  ]
_MINSCALES = [(640, 672, 704, 736, 768, 800), [512], [800]]
_IMSPERBATCH = [2, 4]
_NUMGPUS = 1
_RPN_POSITIVE_RATIOS = [0.33, 0.5]
_GRADIENT_CLIP_VALUE = [1, 5]

_ADD_NOTES = ''

def paramsexist(params):
    #Look in the overview csv file containing all done training runs if random params exists
    csvfile = '/home/althausc/master_thesis_impl/detectron2/out/checkpoints/run_configs.csv'
    with open(csvfile, 'r', newline='') as csvFile:
        reader = csv.reader(csvFile, delimiter='\t')
        content = list(reader)
        header = []
        for i,row in enumerate(content):
            if i==0:
                header = row
                header = [h.strip() for h in header]
            else:
                if row[header.index('NET')] == params['net'] and \
                    eval(row[header.index('Data Augmentation [CropSize, FlipProb, RotationAngle]')]) ==  params['dataaugm'] and \
                    int(row[header.index('ImPerBatch')]) ==  params['batchsize'] and \
                    float(row[header.index('LR')]) ==  params['lr'] and \
                    eval(row[header.index('BN')]) ==  params['bn'] and \
                    int(row[header.index('Min Keypoints')]) ==  params['minkpts'] and \
                    eval(row[header.index('Steps')]) ==  params['steps'] and \
                    float(row[header.index('Gamma')]) ==  params['gamma'] and \
                    eval(row[header.index('MinSize Train')]) ==  params['minscales']:
                    return True
    return False

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
        lr = 0.02/(16/batchsize) #original trained with batchsize=16
        bn = False
        minscales = (640, 672, 704, 736, 768, 800)
        rpn_posratio = 0.5
        gradient_clipvalue = 1  
        minkpts = 1
        steps = [0.76, 0.92] 
        gamma = 0.1

    elif _PARAM_MODE == 'randomsearch':
        while True:
            trainmode = 'ALL'
            dataaugm = random.choice(_DATA_AUGM)
            batchsize = random.choice(_IMSPERBATCH)
            lr = 0.001 #random.choice(_LRS)
            bn = random.choice(_BN)
            minkpts = random.choice(_MINKPTS) #choice?
            step_gam = random.choice(_STEPS_GAMMA)
            steps = step_gam[0]
            gamma = step_gam[1]
            minscales = random.choice(_MINSCALES)
            rpn_posratio = random.choice(_RPN_POSITIVE_RATIOS)
            gradient_clipvalue = 1 
            params = {'net':trainmode, 'dataaugm': dataaugm, 'batchsize': batchsize, 'lr':lr, 'bn':bn,
                    'minkpts':minkpts, 'steps':steps, 'gamma': gamma, 'minscales': minscales}
           
            if not paramsexist(params):
                break    

    elif _PARAM_MODE == 'custom':
        trainmode = 'SCRATCH' #'ALL' #'SCRATCH'
        dataaugm = True
        batchsize = 4
        lr = 0.001 #0.01
        bn = True
        minkpts = 4 
        steps = np.linspace(0.7, 1, 10).tolist()
        gamma = 0.75  #0.75 ^ 10 = 0.05
        minscales = (512,)
        rpn_posratio = 0.5
        gradient_clipvalue = 1

    elif _PARAM_MODE == 'loadparams':
        parser = argparse.ArgumentParser()
        parser.add_argument('-paramsconfig', required=True, help='Load params from file.')
        args = parser.parse_args()

        params = None
        with open(args.paramsconfig, 'r') as f:
            params = json.load(f)

        _ADD_NOTES = 'Rerun of 10-15_19-31-30_all'
        params.update({'addnotes': _ADD_NOTES})

    else:
        raise ValueError()



    print(" --------------------------- RUN %d ------------------------------ "%i)
    if _PARAM_MODE != _PARAM_MODES[4]:
        params = {'trainmode': trainmode, 'dataaugm': dataaugm, 'batchsize': batchsize, 'epochs': _NUMEPOCHS, 'lr': lr, 'bn': bn, 'minscales': minscales, 'rpn_posratio': rpn_posratio,
                'gradient_clipvalue': gradient_clipvalue, 'steps': steps, 'gamma': gamma, 'minkpt': minkpts, 'addnotes': _ADD_NOTES}
    print("Parameters: ",params)
    logdir = os.path.join('/home/althausc/master_thesis_impl/detectron2/out/checkpoints/trainconfigs_tmp', datetime.datetime.now().strftime('%m-%d_%H-%M-%S')+ '_%d'%i)
    os.makedirs(logdir)
    paramconfig = os.path.join(logdir, 'paramsconfig.txt')

    with open(paramconfig, 'w') as f:
        json.dump(params, f)

    #Normal run command, Print for eventually debugging
    gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun_G1d4-1.sh'
    resume = False #True
    maskrcnn_cp = '/home/althausc/master_thesis_impl/detectron2/out/checkpoints/08/07_12-40-41_all/model_0214999.pth'

    #Additional params to setup in file: folder : BN[yes,no], LR, Weight Decay[steps& exp/no exp], Data Augmentation[Random Rotation, Flip & Crop],
    #                                             Min keypoints[for filter images], Additional[MinSizeTrain, ImgPerBatch]

    cmd = "{} python3.6 /home/althausc/master_thesis_impl/scripts/detectron2/MaskRCNN_train_styletransfer.py -paramsconfig {} -addconfig {}"\
                                                    .format(gpu_cmd, paramconfig, '-resume %s'%maskrcnn_cp if resume else ' ')

    print(cmd)

    #Start sbatch training
    gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun1-1.sh'
    jobname = "maskrcnn-train-%s"%datetime.datetime.now().strftime('%d_%H-%M-%S')
    logfile = os.path.join(logdir, 'train.log')

    #Additional params to setup in file: folder : BN[yes,no], LR, Weight Decay[steps& exp/no exp], Data Augmentation[Random Rotation, Flip & Crop],
    #                                             Min keypoints[for filter images], Additional[MinSizeTrain, ImgPerBatch]

    cmd = ("sbatch -w devbox4 -J {} -o {} "+ \
        "{} python3.6 /home/althausc/master_thesis_impl/scripts/detectron2/MaskRCNN_train_styletransfer.py -paramsconfig {} {} -addconfig")\
                                                    .format(jobname, logfile, gpu_cmd, paramconfig, '-resume %s'%maskrcnn_cp if resume else ' ')

    print(cmd)
    os.system(cmd)
    time.sleep(10)
    print()



    """
 print(row[header.index('NET')] == params['net'], \
                    eval(row[header.index('Data Augmentation [CropSize, FlipProb, RotationAngle]')]) ==  params['dataaugm'],\
                    int(row[header.index('ImPerBatch')]) ==  params['batchsize'], \
                    float(row[header.index('LR')]) ==  params['lr'], \
                    eval(row[header.index('BN')]) ==  params['bn'], \
                    int(row[header.index('Min Keypoints')]) ==  params['minkpts'], \
                    eval(row[header.index('Steps')]) ==  params['steps'], \
                    float(row[header.index('Gamma')]) ==  params['gamma'], \
                    eval(row[header.index('MinSize Train')]) ==  params['minscales'] )

                print( eval(row[header.index('Steps')]), params['steps'])
                print( type(eval(row[header.index('Steps')])), type(params['steps']))


                print(row[header.index('NET')] , params['net'], \
                    row[header.index('Data Augmentation [CropSize, FlipProb, RotationAngle]')] , params['dataaugm'],\
                    int(row[header.index('ImPerBatch')]), params['batchsize'], \
                    float(row[header.index('LR')]),  params['lr'], \
                    row[header.index('BN')] ,  params['bn'], \
                    int(row[header.index('Min Keypoints')]), params['minkpts'], \
                    eval(row[header.index('Steps')]), params['steps'], \
                    float(row[header.index('Gamma')]) , params['gamma'], \
                    eval(row[header.index('MinSize Train')]),  params['minscales'])

                print("------------------------------")

    """