from numpy import genfromtxt
import matplotlib.pyplot as plt
import os

import pandas as pd

#Used for plotting coco evaluation results
#Not used now

def plotAPS(output_path):
    bbox_path = os.path.join(output_path,'bbox.csv')
    kps_path = os.path.join(output_path,'keypoints.csv')

    if os.path.exists(bbox_path):
        df=pd.read_csv(bbox_path, sep=',')
        #axs = df.plot(subplots=True, layout=(2,6))
       
        fig = plt.figure()
        plt.title("Bounding Boxes Validation Average Precision")
        df[["AP", "AP50", "AP75", "APs", "APm", "APl"]].plot()
        plt.xlabel('half epochs')
        plt.savefig(os.path.join(output_path,'bbox.png'))

    if os.path.exists(kps_path):
        df=pd.read_csv(kps_path, sep=',')
        #axs = df.plot(subplots=True, layout=(2,6))
        
        fig = plt.figure()
        plt.title("Keypoints Validation Average Precision")
        df[["AP", "AP50", "AP75", "APs", "APm", "APl"]].plot()
        plt.xlabel('half epochs')
        plt.savefig(os.path.join(output_path,'keypoints.png'))

#plotAPS("/home/althausc/master_thesis_impl/detectron2/out/checkpoints/08/03_16-08-48/")