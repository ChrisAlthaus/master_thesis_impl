
import argparse
import os

import datetime
import json
import random
import time

import itertools
import numpy as np
import csv
import sys
sys.path.append('/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch')

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer

import torch


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-config_file', help='Path to the model file.')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    
    args = parser.parse_args()

    val_data_loaders = make_data_loader(
            cfg,
            mode='val',
            is_distributed=False
        )

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.MODEL.LOAD_DATASETSTATS_PATH = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/sgdet_training/12-02_09-23-52-dev3/VG_stanford_filtered_with_attribute_train_statistics.cache'
    cfg.freeze()
    print(cfg)

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    model_dir = cfg.MODEL.PRETRAINED_DETECTOR_CKPT

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=model_dir)
    _ = checkpointer.load()
    
    dataset_name = cfg.DATASETS.VAL

    output_dir = os.path.join(model_dir, '.eval')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        print("Warning: Output directory %s already exists."%output_dir)

    #Writing config to file
    with open(os.path.join(output_dir, 'config.txt'), 'a') as f:
        f.write("Model: %s"%model_dir + os.linesep)
        f.write("Dataset name: %s"%dataset_name + os.linesep)


    valresults = run_val(cfg, dataset_name, model, val_data_loaders[0], False, None)
    savetocsv(valresults, output_dir)



def run_val(cfg, datasetname, model, val_data_loader, distributed, logger):
    torch.cuda.empty_cache()
    iou_types = ("bbox",) #("segm",)("keypoints",)("relations", )("attributes", )

    dataset_results, score = inference(
                                cfg,
                                model,
                                val_data_loader,
                                dataset_name=datasetname,
                                iou_types=iou_types,
                                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                                device=cfg.MODEL.DEVICE,
                                expected_results=cfg.TEST.EXPECTED_RESULTS,
                                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                                output_folder=None,
                                logger=logger,
                    )
    return dataset_results


def savetocsv(val_results, outputdir):
    print("Save following scores to CSV-file:")
    headers = []
    scores = []
    for k, v in val_results['eval_recall'].result_dict['sgdet_recall'].items():
        tag = 'recall/R@{}'.format(k) 
        print(tag , np.mean(v))
        headers.append(tag)
        scores.append(np.mean(v))
    for k, v in val_results['eval_nog_recall'].result_dict['sgdet_recall_nogc'].items():
        tag = 'recall_nogc/R@{}'.format(k)
        print(tag , np.mean(v))
        headers.append(tag)
        scores.append(np.mean(v))
    for k, v in val_results['eval_zeroshot_recall'].result_dict['sgdet_zeroshot_recall'].items():
        tag = 'recall_zero/R@{}'.format(k)
        print(tag , np.mean(v))
        headers.append(tag)
        scores.append(np.mean(v))
    for k, v in val_results['eval_ng_zeroshot_recall'].result_dict['sgdet_ng_zeroshot_recall'].items():
        tag = 'recall_ng_zero/R@{}'.format(k)
        print(tag , np.mean(v))
        headers.append(tag)
        scores.append(np.mean(v))
    for k, v in val_results['eval_mean_recall'].result_dict['sgdet_mean_recall'].items():
        tag = 'recall_mean/R@{}'.format(k)
        print(tag , float(v))
        headers.append(tag)
        scores.append(float(v))
    for k, v in val_results['eval_ng_mean_recall'].result_dict['sgdet_ng_mean_recall'].items():
        tag = 'recall_ng_mean/R@{}'.format(k)
        print(tag , float(v))
        headers.append(tag)
        scores.append(float(v))

    filepath = os.path.join(outputdir, 'evalresults.csv')
    with open(filepath, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(headers)
        writer.writerow(scores)


if __name__ == "__main__":
    main()