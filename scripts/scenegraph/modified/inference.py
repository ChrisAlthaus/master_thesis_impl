#Path: /home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/maskrcnn_benchmark/engine/inference.py
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os
import sys

import json
import torch
from tqdm import tqdm
import math

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug
from maskrcnn_benchmark.structures.bounding_box import BoxList


def compute_on_dataset(model, data_loader, device, synchronize_gather=True, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    torch.cuda.empty_cache()
    flag = True
    for i, batch in enumerate(data_loader):
        with torch.no_grad():
            images, targets, image_ids = batch
            targets = [target.to(device) for target in targets]
            if timer:
                timer.tic()
           
            #print(i, images, "TARGET:", targets, "ID", image_ids)    
            #modified: catch assertion error
            try:
                if cfg.TEST.BBOX_AUG.ENABLED:
                    output = im_detect_bbox_aug(model, images, device)
                else:
                    # relation detection needs the targets
                    output = model(images.to(device), targets)
                #print("OUTPUT: ",output, type(output))

                if timer:
                    if not cfg.MODEL.DEVICE == 'cpu':
                        torch.cuda.synchronize()
                    timer.toc()
                output = [o.to(cpu_device) for o in output] 
                
            except AssertionError as e:
                print("WARNING: ", e)
                output = [None] * len(image_ids) #To assure correct indexing of imageids

        if synchronize_gather:
            synchronize()
            multi_gpu_predictions = all_gather({img_id: result for img_id, result in zip(image_ids, output)})
            if is_main_process():
                for p in multi_gpu_predictions:
                    results_dict.update(p)

        else:
            results_dict.update(
                {img_id: result for img_id, result in zip(image_ids, output)}
            )

        #modified    
        if i%100 == 0:
            print("Processed so far {} images".format(i))
        #modified end
    torch.cuda.empty_cache()
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu, synchronize_gather=True):
    if not synchronize_gather:
        all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return

    if synchronize_gather:
        predictions = predictions_per_gpu
    else:
        # merge the list of dicts
        predictions = {}
        for p in all_predictions:
            predictions.update(p)
    
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "WARNING! WARNING! WARNING! WARNING! WARNING! WARNING!"
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        cfg,
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        logger=None,
):
    print("Start inference")
    load_prediction_from_cache = cfg.TEST.ALLOW_LOAD_FROM_CACHE and output_folder is not None and os.path.exists(os.path.join(output_folder, "eval_results.pytorch"))
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    if logger is None:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    if load_prediction_from_cache:
        predictions = torch.load(os.path.join(output_folder, "eval_results.pytorch"), map_location=torch.device("cpu"))['predictions']
    else:
        predictions = compute_on_dataset(model, data_loader, device, synchronize_gather=cfg.TEST.RELATION.SYNC_GATHER, timer=inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    if not load_prediction_from_cache:
        predictions = _accumulate_predictions_from_multiple_gpus(predictions, synchronize_gather=cfg.TEST.RELATION.SYNC_GATHER)

    if not is_main_process():
        return [], -1.0

    #if output_folder is not None and not load_prediction_from_cache:
    #    torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    if cfg.TEST.CUSTUM_EVAL:
        detected_sgg = custom_sgg_post_precessing(predictions, cfg)
        #modified: save huge number of predictions in multiple files to prevent error
        
        #batchsize = 10000
        #for i in range(0, math.ceil(len(detected_sgg)/batchsize)):
        #    dsplit = {k:detected_sgg[k] for k in list(detected_sgg.keys())[i*batchsize: (i+1)*batchsize]}
        #    with open(os.path.join(cfg.DETECTED_SGG_DIR, 'custom_prediction_{}.json'.format(i)), 'w') as outfile:  
        #        json.dump(dsplit, outfile)
        #        print("Saved custom_prediction_{}.json".format(i))
        with open(os.path.join(cfg.DETECTED_SGG_DIR, 'custom_prediction.json'), 'w') as outfile:  
                json.dump(detected_sgg, outfile)

        #modified end
        print('=====> ' + str(os.path.join(cfg.DETECTED_SGG_DIR, 'custom_prediction.json')) + ' SAVED !')
        return [], -1.0
    print("End inference")
    return evaluate(cfg=cfg,
                    dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    logger=logger,
                    **extra_args)

#modified
filtermod_dir = '/home/althausc/master_thesis_impl/scripts/scenegraph'
sys.path.insert(0,filtermod_dir)
from filter_resultgraphs_modules import get_topkpredictions

sys.path.insert(0,'/home/althausc/master_thesis_impl/scripts/utils')
from statsfunctions import getwhiskersvalues
#modified end

def custom_sgg_post_precessing(predictions, cfg):
    #modified: Postprocessing parameters & stats variable for filtering
    boxes_topk = cfg.TEST.POSTPROCESSING.TOPKBOXES
    rels_topk = cfg.TEST.POSTPROCESSING.TOPKRELS
    filtertresh_boxes = cfg.TEST.POSTPROCESSING.TRESHBOXES
    filtertresh_rels = cfg.TEST.POSTPROCESSING.TRESHRELS

    stats = {'Raw bbox nums': [],
             'Reduced bbox nums': [],
             'Raw rels nums': [],
             'Reduced rels nums': []
    }

    #modified end

    output_dict = {}
    for idx, boxlist in enumerate(predictions):
        #modified
        if boxlist is None:
            continue
        #modified end
        xyxy_bbox = boxlist.convert('xyxy').bbox
        # current sgg info
        current_dict = {}
        # sort bbox based on confidence
        sortedid, id2sorted = get_sorted_bbox_mapping(boxlist.get_field('pred_scores').tolist())
        # sorted bbox label and score
        bbox = []
        bbox_labels = []
        bbox_scores = []
        for i in sortedid:
            bbox.append(xyxy_bbox[i].tolist())
            bbox_labels.append(boxlist.get_field('pred_labels')[i].item())
            bbox_scores.append(boxlist.get_field('pred_scores')[i].item())
        current_dict['bbox'] = bbox
        current_dict['bbox_labels'] = bbox_labels
        current_dict['bbox_scores'] = bbox_scores
        # sorted relationships
        rel_sortedid, _ = get_sorted_bbox_mapping(boxlist.get_field('pred_rel_scores')[:,1:].max(1)[0].tolist())
        # sorted rel
        rel_pairs = []
        rel_labels = []
        rel_scores = []
        rel_all_scores = []
        for i in rel_sortedid:
            rel_labels.append(boxlist.get_field('pred_rel_scores')[i][1:].max(0)[1].item() + 1)
            rel_scores.append(boxlist.get_field('pred_rel_scores')[i][1:].max(0)[0].item())
            rel_all_scores.append(boxlist.get_field('pred_rel_scores')[i].tolist())
            old_pair = boxlist.get_field('rel_pair_idxs')[i].tolist()
            rel_pairs.append([id2sorted[old_pair[0]], id2sorted[old_pair[1]]])
        current_dict['rel_pairs'] = rel_pairs
        current_dict['rel_labels'] = rel_labels
        current_dict['rel_scores'] = rel_scores
        current_dict['rel_all_scores'] = rel_all_scores
        
        #modified: Postprocessing
        #output_dict[idx] = current_dict
        output_dict[idx], statsnum, tstr = get_topkpredictions(current_dict, boxes_topk, rels_topk, filtertresh_boxes, filtertresh_rels)

        stats['Raw bbox nums'].append(statsnum['Raw bbox num'])
        stats['Reduced bbox nums'].append(statsnum['Reduced bbox num'])
        stats['Raw rels nums'].append(statsnum['Raw rels num'])
        stats['Reduced rels nums'].append(statsnum['Reduced rels num'])

        if idx<1000:
            with open(os.path.join(cfg.DETECTED_SGG_DIR, 'stats-singlepred.txt'), 'a') as f:
                f.write(tstr + os.linesep)
        #modified end

    #modified: Save transform statistics
    with open(os.path.join(cfg.DETECTED_SGG_DIR, 'stats.txt'), 'a') as f:
        for name, statslist in stats.items():
            #print(name,statslist)
            f.write(name + os.linesep)
            f.write(getwhiskersvalues(statslist) + os.linesep)
    #modified end
    return output_dict
    
def get_sorted_bbox_mapping(score_list):
    sorted_scoreidx = sorted([(s, i) for i, s in enumerate(score_list)], reverse=True)
    sorted2id = [item[1] for item in sorted_scoreidx]
    id2sorted = [item[1] for item in sorted([(j,i) for i, j in enumerate(sorted2id)])]
    return sorted2id, id2sorted
