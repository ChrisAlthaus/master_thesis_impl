from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm


class LoggingHook(HookBase):
    def __init__(self, cfg, period):
        self._cfg = cfg
        self._period = period
    
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            print("---------------- CONFIGURATION FOR TRAINING MODEL & SOLVER -----------------")
            print(self._cfg)
            print("----------------------------------------------------------------------------")

