from detectron2.engine import DefaultTrainer
from detectron2.evaluation.coco_evaluation import COCOEvaluator
import os
from LossEvalHook import *
from LoggingHook import *
from plotTrainValLosses import saveTrainValPlot


class COCOTrainer(DefaultTrainer):
    def __init__(self,cfg, mode="singlegpu"):
        super().__init__(cfg)
        if mode == "multigpu":
            self.losseval_hook = LossEvalHook(self.cfg,self.model,
                build_detection_test_loader(
                    self.cfg,
                    self.cfg.DATASETS.TEST[0],
                    DatasetMapper(self.cfg,True)
                ))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR,"inference")
        print("--------------------- BUILD EVALUATOR --------------------")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    #For Single-GPU loss computation, uncomment if not used
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(self.cfg,self.model,
                build_detection_test_loader(
                    self.cfg,
                    self.cfg.DATASETS.TEST[0],
                    DatasetMapper(self.cfg,True)
                )) )
        hooks.insert(-1,LoggingHook(self.cfg, self.cfg.TEST.EVAL_PERIOD))
        
        return hooks

    #For Multi-GPU loss computation, uncomment if not used
    """def run_step(self) -> None:
        super().run_step()

        next_iter = self.iter + 1

        is_final = next_iter == self.max_iter
        if is_final or (self.cfg.TEST.EVAL_PERIOD > 0 and next_iter % self.cfg.TEST.EVAL_PERIOD== 0):
            validation_loss = self.losseval_hook._do_loss_eval()
            self.storage.put_scalar('validation_loss', validation_loss)

        if self.cfg.TEST.PLOT_PERIOD != -1 and (next_iter % self.cfg.TEST.PLOT_PERIOD == 0):
            saveTrainValPlot(self.cfg.OUTPUT_DIR)
        
        self.storage.put_scalars(timetest=12)"""