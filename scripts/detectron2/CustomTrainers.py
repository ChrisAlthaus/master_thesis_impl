from detectron2.engine import DefaultTrainer
from detectron2.evaluation.coco_evaluation import COCOEvaluator
import os
from Hooks import LossEvalHook, LoggingHook, EarlyStoppingHook
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper, build_detection_train_loader, build_detection_test_loader
import detectron2.data.detection_utils as utils
from detectron2.engine.hooks import PeriodicWriter
import torch
import copy

#Custom trainer used for additional data augmentations 
#and logging of different metrics (vallidation loss, parameters).
#Additional: Early Stopping

class COCOTrainer(DefaultTrainer):
    def __init__(self,cfg, mode="singlegpu"):
        super().__init__(cfg)
     
    @classmethod
    def build_train_loader(cls, cfg):
        augmentations = []

        #if cfg.INPUT.CROP.ENABLED:
        #    augmentations.append(T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))# not used, since also done by DatasetMapper
        _PROB_HIGH = 0.3
        _PROB_LOW = 0.2

        if cfg.INPUT.RESIZE_SHORTEST_EDGE:
            augmentations.append(T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING))

        if cfg.DATA_AUGMENTATION:
            if cfg.DATA_FLIP_ENABLED:
                augmentations.append(T.RandomFlip(_PROB_LOW, horizontal=True))
            if cfg.ROTATION_ENABLED:
                augmentations.append(T.RandomApply(T.RandomRotation([-15,15]), prob=_PROB_LOW))
            if cfg.COLOR_AUGM_ENABLED:
                augmentations.append(T.RandomApply(transform=T.RandomBrightness(intensity_min=0.75, intensity_max=1.25), prob=_PROB_HIGH))
                augmentations.append(T.RandomApply(transform=T.RandomContrast(intensity_min=0.76, intensity_max=1.25), prob=_PROB_HIGH)) 
                augmentations.append(T.RandomApply(transform=T.RandomSaturation(intensity_min=0.75, intensity_max=1.25), prob=_PROB_HIGH))
            if cfg.INPUT.CROP.ENABLED:
                augmentations.append(T.RandomApply(T.RandomCrop("relative_range", [0.9, 0.9]), prob=_PROB_HIGH))
                #augmentations.append(T.RandomApply(T.RandomCrop("relative_range", [0.9, 0.9]), prob=_PROB_LOW))
                #augmentations.append(T.RandomApply(T.RandomCrop("relative_range", [0.5, 0.2]), prob=_PROB_LOW))


        #return build_detection_train_loader(cfg, mapper=mapper) #test if different result, maybe error in DatasetMapper?!
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations= augmentations))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        #Evaluator used to print & save validation COCO result summary to eventfile
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR,"inference")
        print("--------------------- BUILD EVALUATOR --------------------")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    #For Single-GPU loss computation, uncomment if not used
    def build_hooks(self):
        hooks = super().build_hooks()
        #Hooks:
        # -LossEvalHook: calculate validation loss & save to eventfile
        # -LoggingHook: print config & save specific config parameters to eventfile
        # -EarlyStoppingHook: stop training if loss of windowsizes don't decrease

        #Insert hooks before PeriodicWriter at last position
        hooks.insert(-1,LossEvalHook(self.cfg.TEST.EVAL_PERIOD,self.model,
                build_detection_test_loader(
                    self.cfg,
                    self.cfg.DATASETS.TEST[0],
                    DatasetMapper(self.cfg,True)
                ),
                self.cfg.TEST.PLOT_PERIOD, self.cfg.OUTPUT_DIR))
        hooks.insert(-1,LoggingHook(self.cfg))
        hooks.insert(-1,EarlyStoppingHook(self.cfg))

        print("Registered hooks: ",hooks)
        return hooks

    