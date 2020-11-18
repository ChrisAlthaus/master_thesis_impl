#Path: /home/althausc/master_thesis_impl/PoseFix_RELEASE/main/train.py
import tensorflow as tf
import argparse
import numpy as np

from model import Model
from config import cfg
from tfflat.base import Trainer
from tfflat.utils import mem_info

import os
import datetime
import csv
import json

def parse_args():
    parser = argparse.ArgumentParser()
    #modified
    parser.add_argument('--gpu', type=str, dest='gpu_ids') #not used
    parser.add_argument('--continue', dest='continue_train', action='store_true') #not used
    parser.add_argument('--inputpreds', type=str) #not used
    parser.add_argument('--pretrained', type=str) #not used
    parser.add_argument('--paramsconfig', type=str)
    #modified end

    args = parser.parse_args()

    if not args.gpu_ids:
        args.gpu_ids = str(np.argmin(mem_info()))

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args
args = parse_args()

#cfg.set_args(args.gpu_ids, args.inputpreds, args.continue_train)

#modified
cfg.model_dump_dir = os.path.join(cfg.model_dump_dir, datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
if not os.path.exists(cfg.model_dump_dir):
    os.makedirs(cfg.model_dump_dir)
else:
    raise ValueError("Output directory %s for checkpoints already exists. Please wait a few minutes."%cfg.model_dump_dir)
print("Model output directory: ", cfg.model_dump_dir)

#modified: to support tuple and list arguments
print("Reading config (hyper-)parameters from file: ",args.paramsconfig)
params = []
with open (args.paramsconfig, "r") as f:
    params = json.load(f)

cfg.batch_size= params['trainbatchsize']
cfg.end_epoch = params['epochs']
cfg.lr_dec_epoch = params['steps_epochs']
cfg.lr_dec_factor = params['lr_dec_factor']
cfg.lr = params['lr']
cfg.predict_inputmodel_file = params['predictionsfile']

if params['resumecpkt']:
    cfg.model_pretrained_dir = params['resumecpkt']
    cfg.continue_train = True
else:
    cfg.model_pretrained_dir = '/home/althausc/master_thesis_impl/PoseFix_RELEASE/output/model_dump/COCO/MSCOCO-pretrained'
#modified end


# Save the entire config to file
cfg.saveconfig(cfg.model_dump_dir, params_train=[args.inputpreds, args.continue_train])
cfg.display = 100 #display & save training stats
cfg.multi_thread_enable = False #True
cfg.num_thread = 4

#Save config row to summary/overview file
filepath = '/home/althausc/master_thesis_impl/PoseFix_RELEASE/output/model_dump/COCO/run_configs.csv'
if not os.path.exists(filepath):
    with open(filepath, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        headers = ['Folder', 'LR', 'BN', 'BatchSize', 'PretrainedModel', 'Epochs', 'LrDecayEpochs', 'LrDecayFactor', 'Loss_h', 'Loss_c', 'Add.Notes']
        writer.writerow(headers)
folder = os.path.basename(cfg.model_dump_dir)
pretrained = cfg.model_pretrained_dir if args.continue_train else 'False'
epochs = cfg.end_epoch
row = [folder, cfg.lr, cfg.bn_train, cfg.batch_size, pretrained, epochs, cfg.lr_dec_epoch, cfg.lr_dec_factor, ' ', ' ', '______']
with open(filepath, 'a') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(row)
print("Sucessfully wrote hyper-parameter row to configs file.")
#modified end



trainer = Trainer(Model(), cfg)
trainer.train()





