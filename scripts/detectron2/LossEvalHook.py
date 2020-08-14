from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import torch
import time
import datetime
import logging
import numpy as np
import os
import cv2
import csv

from plotTrainValLosses import saveTrainValPlot

class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader, plot_period, plot_folder):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
        self._plot_period = plot_period
        self._plot_folder = plot_folder
    
    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
           
        mean_loss = np.mean(losses)
        if comm.is_main_process():
            self.trainer.storage.put_scalar('validation_loss', mean_loss)
        #comm.synchronize()

        return mean_loss
            
    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        #print("Calculating loss of data of length: ", len(data))
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
        
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            validation_loss = self._do_loss_eval()
            
            """
            #Write validation losses additional to file, because metrics.json sometimes not updating
            filepath = os.path.join(self._plot_folder, 'validation_losses.csv')
                
            if not os.path.exists(filepath):
                columns = ["iter","validation_loss" ]
                wtr = csv.writer(open (filepath, 'a'), delimiter=',', lineterminator=os.linesep)
                wtr.writerow(columns)
            
            wtr = csv.writer(open (filepath, 'a'), delimiter=',', lineterminator=os.linesep)
            wtr.writerow([self.trainer.iter,validation_loss])"""
        """
        #Save plot of train & validation loss and write to tensorboard    
        if self._plot_period != -1 and (next_iter % self._plot_period == 0):
            if comm.is_main_process():
                img_path = saveTrainValPlot(self._plot_folder)
                #img = cv2.imread(img_path)
                #self.trainer.storage.put_image(os.path.basename(img_path),img)
                #comm.synchronize()
                #self.trainer.storage.clear_images()
            comm.synchronize()
        self.trainer.storage.put_scalars(timetest=12)
        """