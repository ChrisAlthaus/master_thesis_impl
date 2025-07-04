#Path: /home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/tools/relation_test_net.py
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import datetime

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

# Check if we can enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for mixed precision via apex.amp')

#sbatch -w devbox4 -J graphprediction -o /home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/out/predictions/graphs/logs/02-11_18-00-04.txt /home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun1-2.sh python3.6 -m torch.distributed.launch     --master_port 10056     --nproc_per_node=1 /home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/tools/relation_test_net.py    --config-file "/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/configs/e2e_relation_X_101_32_8_FPN_1x.yaml"     MODEL.ROI_RELATION_HEAD.USE_GT_BOX False        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False        MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor       MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TE  MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum  MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs     TEST.IMS_PER_BATCH 1     DTYPE "float16"         GLOVE_DIR /home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/sgdet_training/glove      MODEL.PRETRAINED_DETECTOR_CKPT /home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/sgdet_training/12-02_09-23-52-dev3   OUTPUT_DIR /home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/sgdet_training/12-02_09-23-52-dev3        TEST.CUSTUM_EVAL True   TEST.CUSTUM_PATH /nfs/data/iart/art500k/img       TEST.POSTPROCESSING.TOPKBOXES -1        TEST.POSTPROCESSING.TOPKRELS 50  TEST.POSTPROCESSING.TRESHBOXES 0.1    TEST.POSTPROCESSING.TRESHRELS 0.1       DETECTED_SGG_DIR /home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/out/predictions/graphs

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    #modified
    parser.add_argument('-topkboxes', type=int, default=20,
                    help='Filter the predictions and take best k boxes.')
    parser.add_argument('-topkrels', type=int, default=40,
                        help='Filter the predictions and take best k relations.')
    parser.add_argument('-filtertresh_boxes', type=float, default=0.5,
                        help='Only boxes above this confidence treshold will be considered.')
    parser.add_argument('-filtertresh_rels', type=float, default=0.5,
                        help='Only relations above this confidence treshold will be considered.')
    #modified end

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.MODEL.DEVICE = 'cpu' #'cuda' #'cpu'

    #modified: config for postprocessing & prevent from loading dataset (costly)   
    #cfg.TEST.POSTPROCESSING.TOPKBOXES = 100
    #fg.TEST.POSTPROCESSING.TOPKRELS = 100
    cfg.TEST.POSTPROCESSING.TRESHBOXES = 0.001
    cfg.TEST.POSTPROCESSING.TRESHRELS = 0.001
    #set to '' if new vgdataset (train set)
    cfg.MODEL.LOAD_DATASETSTATS_PATH = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/sgdet_training/12-02_09-23-52-dev3/VG_stanford_filtered_with_attribute_train_statistics.cache'
    #modified end

    print(args.opts)
    cfg.merge_from_list(args.opts)
    
    cfg.freeze()
    print(cfg)

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    # Initialize mixed-precision if necessary
    use_mixed_precision = cfg.DTYPE == 'float16'
    amp_handle = amp.init(enabled=use_mixed_precision, verbose=cfg.AMP_VERBOSE)

    output_dir = cfg.OUTPUT_DIR

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

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

    #modified: Create output directory & Save config/flags
    cfg.defrost()
    cfg.DETECTED_SGG_DIR = os.path.join(cfg.DETECTED_SGG_DIR, datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
    if not os.path.exists(cfg.DETECTED_SGG_DIR):
        os.makedirs(cfg.DETECTED_SGG_DIR)
    else:
        raise ValueError("Output directory %s already exists."%cfg.DETECTED_SGG_DIR)
    cfg.freeze()

    #Writing config to file
    with open(os.path.join(cfg.DETECTED_SGG_DIR, 'config.txt'), 'a') as f:
        f.write("Input image directory: %s"%cfg.TEST.CUSTUM_PATH + os.linesep)
        f.write("Backbone used: %s"%cfg.MODEL.PRETRAINED_DETECTOR_CKPT + os.linesep)
        f.write("Model used: %s"%cfg.OUTPUT_DIR + os.linesep)
        f.write("Topk Boxes: %d"%cfg.TEST.POSTPROCESSING.TOPKBOXES + os.linesep)
        f.write("Topk Rels: %d"%cfg.TEST.POSTPROCESSING.TOPKRELS + os.linesep)
        f.write("Filter treshold Boxes: %f"%cfg.TEST.POSTPROCESSING.TRESHBOXES + os.linesep)
        f.write("Filter treshold Rels: %f"%cfg.TEST.POSTPROCESSING.TRESHRELS + os.linesep)

    #modified end

    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
            
    data_loaders_val = make_data_loader(cfg, mode="test", is_distributed=distributed)
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


if __name__ == "__main__":
    main()
