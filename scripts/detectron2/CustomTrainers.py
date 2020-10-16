from detectron2.engine import DefaultTrainer
from detectron2.evaluation.coco_evaluation import COCOEvaluator
import os
from Hooks import LossEvalHook, LoggingHook, EarlyStoppingHook
from plotTrainValLosses import saveTrainValPlot
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper, build_detection_train_loader, build_detection_test_loader
import detectron2.data.detection_utils as utils
import torch
import copy


class COCOTrainer(DefaultTrainer):
    def __init__(self,cfg, mode="singlegpu"):
        super().__init__(cfg)
        """if mode == "multigpu":
            self.losseval_hook = LossEvalHook(self.cfg.TEST.EVAL_PERIOD,self.model,
                build_detection_test_loader(
                    self.cfg,
                    self.cfg.DATASETS.TEST[0],
                    DatasetMapper(self.cfg,True)
                ),
                self.cfg.TEST.PLOT_PERIOD,self.cfg.OUTPUT_DIR)""" # deprecated?
     
    @classmethod
    def build_train_loader(cls, cfg):
        augmentations = []

        #if cfg.INPUT.CROP.ENABLED:
        #    augmentations.append(T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))# not used, since also done by DatasetMapper
        if cfg.DATA_FLIP_ENABLED:
            augmentations.append(T.RandomFlip(cfg.DATA_FLIP_PROBABILITY, horizontal=True))
        if cfg.ROTATION_ENABLED:
            augmentations.append(T.RandomRotation(cfg.ROTATION))

        def mapper(dataset_dict):
            # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
            dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
            image = utils.read_image(dataset_dict["file_name"], format="BGR")

            augmentations = []
            augmentations.append([T.ResizeShortestEdge((640, 672, 704, 736, 768, 800), 1333, 'choice')])
            augmentations.append(T.RandomCrop(0.9, 0.9))# not used, since also done by DatasetMapper
            augmentations.append(T.RandomFlip(0.25, horizontal=True))
            augmentations.append(T.RandomRotation([-15,15]))


            """augmentations.append(T.RandomApply(transform=T.RandomBrightness(intensity_min=0.75, intensity_max=1.25),
                            prob=0.20))
            augmentations.append(T.RandomApply(transform=T.RandomContrast(intensity_min=0.76, intensity_max=1.25),
                            prob=0.20)) 
            augmentations.append(T.RandomApply(transform=T.RandomSaturation(intensity_min=0.75, intensity_max=1.25)))"""

            image, transforms = T.apply_transform_gens([augmentations], image)

            dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

            annos = [
                utils.transform_instance_annotations(obj, transforms, image.shape[:2])
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image.shape[:2])
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
            return dataset_dict

        #return build_detection_train_loader(cfg, mapper=mapper) #test if different result, maybe error in DatasetMapper?!
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations= augmentations))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR,"inference")
        print("--------------------- BUILD EVALUATOR --------------------")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    #For Single-GPU loss computation, uncomment if not used
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(self.cfg.TEST.EVAL_PERIOD,self.model,
                build_detection_test_loader(
                    self.cfg,
                    self.cfg.DATASETS.TEST[0],
                    DatasetMapper(self.cfg,True)
                ),
                self.cfg.TEST.PLOT_PERIOD,self.cfg.OUTPUT_DIR))
        hooks.insert(-1,LoggingHook(self.cfg, self.cfg.TEST.EVAL_PERIOD))
        hooks.insert(-1,EarlyStoppingHook(self.cfg))
        
        return hooks

    