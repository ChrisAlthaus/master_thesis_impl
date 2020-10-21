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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    parser.add_argument('--inputpreds', type=str)
    parser.add_argument('--pretrained', type=str)

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

cfg.set_args(args.gpu_ids, args.inputpreds, args.continue_train)

#modified
cfg.model_dump_dir = os.path.join(cfg.model_dump_dir, datetime.datetime.now().strftime('%m/%H-%M-%S'))
if not os.path.exists(cfg.model_dump_dir):
    os.makedirs(cfg.model_dump_dir)
else:
    raise ValueError("Output directory %s for checkpoints already exists. Please wait a few minutes."%cfg.model_dump_dir)

#Pretrained checkpoints
if args.continue_train:
    cfg.model_pretrained_dir = args.pretrained
else: # only for ResNet152
    cfg.model_pretrained_dir = '/home/althausc/master_thesis_impl/PoseFix_RELEASE/output/model_dump/COCO/MSCOCO-pretrained'

# Save the entire config to file
cfg.saveconfig(cfg.model_dump_dir, params_train=[args.inputpreds, args.continue_train])
cfg.display = 100 #display & save training stats

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





