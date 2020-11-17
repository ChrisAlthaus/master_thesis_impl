# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import time
import datetime

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, all_gather, is_main_process
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from maskrcnn_benchmark.utils.metric_logger import MetricLogger

from torch.utils.tensorboard import SummaryWriter
import json
import csv
import numpy as np
import cv2
from torchvision import transforms
import sys

from tempfile import NamedTemporaryFile
import shutil

sys.path.append('/home/althausc/master_thesis_impl/scripts/scenegraph')
from visualizeimgs import draw_image

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')


def train(cfg, local_rank, distributed, logger):
    model = build_detection_model(cfg)

    #modified: Save model architecture & layer specs to file 
    if is_main_process():
        print("Save model's architecture:")
        with open(os.path.join(cfg.OUTPUT_DIR, 'model_architectur.txt'), 'w') as f:
            print(list(model.children()),file=f)

        print("Save model's state_dict:")
        with open(os.path.join(cfg.OUTPUT_DIR, 'layer_params_overview.txt'), 'w') as f:
            for name, param in list(model.named_parameters()):
                f.write('{} requires_gradient: {}'.format(name, param.requires_grad)+ os.linesep)
    #modified end

    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model, logger, rl_factor=float(cfg.SOLVER.IMS_PER_BATCH))
    scheduler = make_lr_scheduler(cfg, optimizer)

    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, update_schedule=cfg.SOLVER.UPDATE_SCHEDULE_DURING_LOAD)
    arguments.update(extra_checkpoint_data)
    print("Arguments: ",arguments)
    if cfg.MODEL.RESUME_TRAIN:
        print("Resetting iteration number...")
        arguments["iteration"] = 0
    print("Arguments: ",arguments)

    #modified: important class for data loading: Scene-Graph-Benchmark.pytorch/build/lib.linux-x86_64-3.6/maskrcnn_benchmark/data/datasets/visual_genome.py
    train_data_loader = make_data_loader(
        cfg,
        mode='train',
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )
    val_data_loaders = make_data_loader(
        cfg,
        mode='val',
        is_distributed=distributed,
    )
    #Validation on other dataset possible
    if cfg.DATASETS.VAL2:
        val_data_loaders2 = make_data_loader(
            cfg,
            mode='val2',
            is_distributed=distributed,
        )
    #modified end
    
    #modified
    writer = None
    if is_main_process():
        writer = SummaryWriter(cfg.OUTPUT_DIR)
    c_target_empty = 0
    #modified end

    #modified: illustrate annotations for some validation images
    def check_valannotations(dataloader, writer):
        print("Draw annotations for some sample validation images...")
        for i, (images, targets, _) in enumerate(dataloader):
            if i%1000 == 0:
                imgwithann = getfirstimg_withann(images, targets)
                writer.add_image('Validation Sample %d'%i, imgwithann, global_step=i, dataformats='HWC')
                cv2.imwrite(os.path.join(cfg.OUTPUT_DIR, '.images', "validation_sample_%d.jpg"%i), imgwithann)
                print("\tWrote image to tensorboard")
        print("Draw annotations for some sample validation images done.")

    if is_main_process():
        print("Val dataloaders: ",val_data_loaders)
        if not os.path.exists(os.path.join(cfg.OUTPUT_DIR,'.images')):
            os.makedirs(os.path.join(cfg.OUTPUT_DIR,'.images'))
        check_valannotations(val_data_loaders[0], writer)
    #modified end

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    if cfg.SOLVER.PRE_VAL:
        logger.info("Validate before training")
        val_results, _ = run_val(cfg, cfg.DATASETS.VAL, model, val_data_loaders, distributed)
        savevalresult(val_results, arguments["iteration"], cfg.OUTPUT_DIR, writer)
    
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(train_data_loader)
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()
    print("Max Iterations: ",max_iter)

    for iteration, (images, targets, _) in enumerate(train_data_loader, start_iter):
        model.train()
        #modified: don't stop when seeing a target without a box annotation (very rare)
        if any(len(target) < 1 for target in targets):
            logger.error(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
            print(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}")

            inds = [k for k,target in enumerate(targets) if len(target)<1]
            inds.sort(reverse=True)
            print("Deleting indices: ",inds)
            print('Images before: ', images)
            print('Targets before: ', targets)
            targets = list(targets)
            for k in inds:
                images.tensors = torch.cat([images.tensors[0:k], images.tensors[k+1:]])
                del images.image_sizes[k]
                del targets[k]
            targets =tuple(targets)
            print('Images result: ', images)
            print('Targets result: ', targets)
            c_target_empty = c_target_empty + len(inds)
        #modified end
        #modified: log of images to tensorboard
        #Set cfg parameter in Scene-Graph-Benchmark.pytorch/maskrcnn_benchmark/config/defaults.py
        if iteration % cfg.SOLVER.IMG_AUGM_LOGPERIOD == 0 and is_main_process():
            imgwithann = getfirstimg_withann(images, targets)
            writer.add_image('Iteration %d Sample'%iteration, imgwithann, global_step=iteration, dataformats='HWC')
            print("Wrote image to tensorboard with imageid: ",_)
        #modified end

        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        # MATTHIAS CHANGES
        #print(f'Train: process {local_rank}:{iteration}')

        if iteration % 200 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

            #modified: log meters to tensorboard
            if is_main_process():
                print("added loss:", meters.loss.median, iteration)
                writer.add_scalar('loss', meters.loss.median, iteration)
                writer.add_scalar('loss_box_reg', meters.loss_box_reg.median, iteration)
                writer.add_scalar('loss_classifier', meters.loss_classifier.median, iteration)
                writer.add_scalar('loss_objectness', meters.loss_objectness.median, iteration)
                writer.add_scalar('loss_rpn_box_reg', meters.loss_rpn_box_reg.median, iteration)
                writer.add_scalar('lr', optimizer.param_groups[0]["lr"], iteration)
            #modified end

            #modified: add metrics.dat for better summary of training process & read when finished
            if is_main_process():
                with open(os.path.join(cfg.OUTPUT_DIR, "metrics.dat"),"a+") as f:
                    logline = {'iter': iteration}
                    for name, meter in meters.meters.items():
                        logline[name] = meter.median  
                    logline['lr'] = optimizer.param_groups[0]["lr"]  
                    json.dump(logline, f)
                    f.write(os.linesep)
                    #f.write(meters_str + os.linesep)
            #modified end

        if cfg.SOLVER.TO_VAL and iteration % cfg.SOLVER.VAL_PERIOD == 0:
            logger.info("Start validating")
            val_result, _ = run_val(cfg, cfg.DATASETS.VAL, model, val_data_loaders, distributed)
            logger.info("Validation Result: {}".format(str(val_result)))
            savevalresult(val_result, iteration, cfg.OUTPUT_DIR, writer)

        if cfg.DATASETS.VAL2 and cfg.SOLVER.TO_VAL and iteration % cfg.SOLVER.VAL_PERIOD * 2 == 0:
            logger.info("Start validating")
            val_result , _ = run_val(cfg, cfg.DATASETS.VAL2, model, val_data_loaders2, distributed)
            logger.info("Validation Result: {}".format(str(val_result)))
            savevalresult(val_result, iteration, cfg.OUTPUT_DIR, writer, suffix='(2)')

        if iteration % checkpoint_period == 0 and is_main_process(): #modified
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    #modified
    print("Number of empty targets, which were skipped: ", c_target_empty)
    #modified end
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

    return model


def run_val(cfg, datasetname, model, val_data_loaders, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations", )
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes", )
        
    dataset_names = datasetname
    val_results = []
    for dataset_name, val_data_loader in zip(dataset_names, val_data_loaders):
        dataset_results = inference(
                            cfg,
                            model,
                            val_data_loader,
                            dataset_name=dataset_name,
                            iou_types=iou_types,
                            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                            device=cfg.MODEL.DEVICE,
                            expected_results=cfg.TEST.EXPECTED_RESULTS,
                            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                            output_folder=None,
                        )
        print(dataset_results)
        synchronize()  
        val_results.append(dataset_results)
    #modified
    if not is_main_process():
        return [], -1 #only main process used for logging results    
    #gathered_result = all_gather(torch.tensor(dataset_results).cpu())
    #gathered_result = [t.view(-1) for t in gathered_result]
    #gathered_result = torch.cat(gathered_result, dim=-1).view(-1)
    #val_result = gathered_result[gathered_result>=0].numpy()
    #val_result = float(valid_result.mean())
    #del gathered_result#, valid_result
    torch.cuda.empty_cache()
    return dataset_results, score
    #modified end

def run_test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations", )
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes", )
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(
        cfg, 
        mode='test', 
        is_distributed=distributed
        )
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            cfg,
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    
    #modified
    parser.add_argument(
        "--paramsconfig",
        default="",
        help="path to parameters file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    #modified end

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)

    #modified: to support tuple and list arguments
    print("Reading config (hyper-)parameters from file: ",args.paramsconfig)
    params = []
    with open (args.paramsconfig, "r") as f:
        params = json.load(f)

    cfg.SOLVER.IMS_PER_BATCH = params['trainbatchsize']
    cfg.TEST.IMS_PER_BATCH = params['testbatchsize']
    cfg.DTYPE = params['dtype']
    #cfg.SOLVER.BASE_LR = params['lr']
    cfg.SOLVER.MAX_ITER = params['maxiterations']
    cfg.SOLVER.STEPS = tuple(params['steps'])
    cfg.SOLVER.VAL_PERIOD = params['valperiod']
    cfg.SOLVER.CHECKPOINT_PERIOD = params['cpktperiod']
    cfg.DATASETS.SELECT = params['datasetselect']
    cfg.MODEL.RELATION_ON = params['relationon']
    cfg.SOLVER.PRE_VAL = params['preval']
    cfg.OUTPUT_DIR = params['outputdir']

    if params['resumecpkt']:
        cfg.MODEL.RESUME_TRAIN = True
        cfg.MODEL.WEIGHT = params['resumecpkt']

    #Custom arguments setup
    print("Optional arguments: ",args.opts)
    cfg.merge_from_list(args.opts)
    #modified end

    #modified: Set configs
    #cfg.SOLVER.GAMMA = 0.1 #0.3162 #0.1
    #cfg.SOLVER.STEPS = (30000,) #(30000,40000) #(30000,)
    #cfg.MODEL.WEIGHT = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/faster_rcnn_training/11-13_12-44-52/model_final.pth'

    #Entire backbone should be unfreezed
    cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT = 0

    if cfg.DATASETS.SELECT == 'trainandval-subset': 
        #both train & validation annotations are subset of original annotations
        cfg.DATASETS.TRAIN = ("VG_styletransfer_subset_train",)
        cfg.DATASETS.TEST = ("VG_styletransfer_subset_test",)
        cfg.DATASETS.VAL = ("VG_styletransfer_subset_val",) 
    elif cfg.DATASETS.SELECT == 'val-subset': 
        #validation annotations are subset of original annotations  
        cfg.DATASETS.TRAIN = ("VG_styletransfer_val_subset_train",)
        cfg.DATASETS.TEST = ("VG_styletransfer_val_subset_test",)
        cfg.DATASETS.VAL = ("VG_styletransfer_val_subset_val",)
        #additional: compare with unreduced validation set annotations
        cfg.DATASETS.VAL2 = ("VG_styletransfer_val",) 
    elif cfg.DATASETS.SELECT == 'default-styletransfer':
        #no subset filtering   
        cfg.DATASETS.TRAIN = ("VG_styletransfer_train",)
        cfg.DATASETS.TEST = ("VG_styletransfer_test",)
        cfg.DATASETS.VAL = ("VG_styletransfer_val",)
    elif cfg.DATASETS.SELECT == 'default-vg':
        cfg.DATASETS.TRAIN = ("VG_stanford_filtered_with_attribute_train",)
        cfg.DATASETS.TEST = ("VG_stanford_filtered_with_attribute_test",)
        cfg.DATASETS.VAL = ("VG_stanford_filtered_with_attribute_val",)
        #additional: compare with unreduced style-transfer validation set annotations
        cfg.DATASETS.VAL2 = ("VG_styletransfer_val",)
    else: 
        raise ValueError()
    #modified end

    #modified: create for each run a new directory
    output_dir = os.path.join(cfg.OUTPUT_DIR, datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
    if is_main_process():
        os.makedirs(output_dir)
        print("Successfully created output directory: ", output_dir)
    cfg.OUTPUT_DIR = output_dir
    cfg.freeze()
    #modified end

    #modified: add line to overall config file for this run
    if is_main_process():
        save_modelconfigs(os.path.dirname(cfg.OUTPUT_DIR), cfg)
    #modified

    cfg.freeze()

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    if is_main_process():
        save_config(cfg, output_config_path)

    model = train(cfg, args.local_rank, args.distributed, logger)

    if not args.skip_test:
        run_test(cfg, model, args.distributed)

    #modified: deprecated
    #if is_main_process():
    #    add_evalstats(cfg.OUTPUT_DIR)
    #modified end

def savevalresult(val_results, iteration, outputdir, writer, suffix=None):
    #log to tensorboard
    print("save: ",val_results)
    scalars = ['bbox/AP', 'bbox/AP50', 'bbox/AP75', 'bbox/APl', 'bbox/APm', 'bbox/APs']
    scalars = [s if suffix is None else s+suffix for s in scalars]
    if is_main_process():
        for i,scalar in enumerate(scalars):
            writer.add_scalar(scalar, val_results[i], iteration)    
            print("Added: ", scalar, val_results[i], iteration)
        writer.add_scalar('R@100' if suffix is None else 'R@100'+suffix, val_results[0], iteration)  


def getfirstimg_withann(images, targets):
    bboxes = targets[0].bbox
    print("Img tensor size: ",images.tensors[0].size())
    #Shape (C,H,W) -> (H,W,C) ?!
    image = images.tensors[0]
    print("Min of Image Tensor: ",torch.min(image))
    print("Max of Image Tensor: ",torch.max(image))
    image = transforms.ToPILImage()(image).convert("RGB")
    print('Img loaded size: ',image.size)
    #image.save(os.path.join(cfg.OUTPUT_DIR,'test.jpg'))
    ##BGR -> RGB ?!
    blabels = targets[0].extra_fields['labels'].numpy()
    relations = targets[0].extra_fields['relation'].numpy()
    relpairs = []
    rellabels = []
    for k,row in enumerate(relations):
        for l,predicate in enumerate(row):
            if predicate!=0:
                relpairs.append([k,l])
                rellabels.append(predicate)
    print("relpairs ",relpairs)
    print("rellabels ",rellabels)

    img , _ = draw_image(image, bboxes, blabels, relpairs, rellabels)
    img = np.array(img)
    print("Result/Drawn img size: ",img.shape)
    print("Min of Image Result: ",np.min(img))
    print("Max of Image Result: ",np.max(img))
    #img.save(os.path.join(cfg.OUTPUT_DIR,'test2.jpg'))
    return img


def save_modelconfigs(configdir_all, cfg):
    filepath = os.path.join(configdir_all, 'run_configs.csv')

    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            headers = ['Folder', 'Predictor', 'Fusion', 'ContextLayer', 'FasterRCNN', 'Batchsize', 'LR', 'MaxIter', 'ValPeriod',
                       'CpktPeriod', 'Steps', 'Gamma', 'MinSize', 'Dataset', 'Attributes', 'Train Loss', 'R@100']
            writer.writerow(headers)
            
    folder = os.path.basename(cfg.OUTPUT_DIR)
    predictor = cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR
    fusiontype = cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE if predictor == 'CausalAnalysisPredictor' else 'not used'
    contextlayer = cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER if predictor == 'CausalAnalysisPredictor' else 'not used'
    fasterrcnn_dir = os.path.basename(cfg.MODEL.PRETRAINED_DETECTOR_CKPT)
    batchsize = cfg.SOLVER.IMS_PER_BATCH
    lr = cfg.SOLVER.BASE_LR
    maxiter = cfg.SOLVER.MAX_ITER
    valperiod = cfg.SOLVER.VAL_PERIOD
    cpktperiod = cfg.SOLVER.CHECKPOINT_PERIOD
    minsize = cfg.INPUT.MIN_SIZE_TRAIN
    steps = cfg.SOLVER.STEPS
    gamma = cfg.SOLVER.GAMMA
    dataset = cfg.DATASETS.SELECT
    attributes = cfg.MODEL.ATTRIBUTE_ON

    row = [folder, predictor, fusiontype, contextlayer, fasterrcnn_dir, batchsize, lr, maxiter, valperiod,
           cpktperiod, steps, gamma, minsize, dataset, attributes, ' ', ' ']
    with open(filepath, 'a') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(row)
    print("Sucessfully wrote hyper-parameter row to configs file.")

def add_evalstats(modeldir):
    #Get Losses
    lines = []
    with open(os.path.join(modeldir, 'metrics.dat'), 'r') as f:
        for line in f:
            lines.append(eval(line))

    _LASTN = 10
    _SEARCH_LASTN = 100
    trainloss_lastn = [entry['loss'] for entry in lines[-_LASTN:] if 'loss' in entry]
    recall_lastn = [entry['R@100'] for entry in lines[-_SEARCH_LASTN:] if 'R@100' in entry]

    trainloss = np.mean(trainloss_lastn) if len(trainloss_lastn)>0 else 'not found'
    recall = recall_lastn[-1] if len(recall_lastn)>0 else 'not found'
    print("Averaged last N losses:")
    print("\tTrain Loss: ",trainloss)
    print("R@100: ", recall)

    #Update CSV config
    csvfile = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/run_configs.csv'
    tempfile = NamedTemporaryFile('w+t', newline='', delete=False, dir='/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/tmp')
    shutil.copyfile(csvfile, tempfile.name)
    foldername = os.path.basename(modeldir)
    content = None
    with open(csvfile, 'r', newline='') as csvFile:
        reader = csv.reader(csvFile, delimiter='\t')
        content = list(reader)
        for i,row in enumerate(content):
            if i==0:
                header = row
                header = [h.strip() for h in header]
                print(header)
            else:
                #update losses in row of current model entry
                print(row[header.index('Folder')], foldername, row[header.index('Folder')] == foldername)
                if row[header.index('Folder')] == foldername:
                    row[header.index('Train Loss')] = '%.2f'%trainloss if not isinstance(trainloss, str) else trainloss
                    row[header.index('R@100')] = '%.2f'%recall if not isinstance(recall,str) else recall
                    break

    with open(csvfile, 'w', newline='') as csvFile:  
        writer = csv.writer(csvFile, delimiter='\t')
        writer.writerows(content)

if __name__ == "__main__":
    main()
