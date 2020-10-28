import json
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter

import argparse

#deprecated: not used

def saveTrainValPlot(experiment_folder, onlySmoothed = True):
    def load_json_arr(json_path):
        lines = []
        with open(json_path, 'r') as f:
            for line in f:
                lines.append(json.loads(line))
        return lines

    experiment_metrics = load_json_arr(experiment_folder + '/metrics.json')
    plt.rcParams['agg.path.chunksize'] = 10000
    
    #Expand figure width for every 100 datapoints
    figsize = [6.4, 4.8]    #defaults
    figsize[0] = figsize[0] * max(1,min(200,len(experiment_metrics)/100))
    
    x_total = [x['iteration'] for x in experiment_metrics]
    y_total = [x['total_loss'] for x in experiment_metrics]
    x_val = [x['iteration'] for x in experiment_metrics if 'validation_loss' in x]
    y_val = [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x]
    
    plt.figure(figsize=tuple(figsize))
    
    if not onlySmoothed:
        plt.plot(x_total, y_total, label='total_loss')
        plt.plot(x_val, y_val, label='validation_loss')

        #print("total losses: ",y_total)
        print("total losses size: ",len(y_total))
        #print("val losses: ",y_val)
        print("val losses size: ",len(y_val))

    if len(y_total) > 10:
        w_size = int(len(y_total)/2)    #window size must be odd
        w_size = w_size - 1 if w_size % 2 == 0 else w_size
        y2_total_smoothed = savgol_filter(y_total, w_size, min(3,w_size-1))
        plt.plot(x_total, y2_total_smoothed, label='total_loss_smoothed')

    if len(y_val) > 10:
        w_size = int(len(y_val)/2)    #window size must be odd
        w_size = w_size - 1 if w_size % 2 == 0 else w_size
        y2_val_smoothed = savgol_filter(y_val, w_size, min(3,w_size-1))
        plt.plot(x_val, y2_val_smoothed, label='val_loss_smoothed')
    

    plt.legend(loc='upper left')
    plt.savefig(os.path.join(experiment_folder, str(experiment_metrics[-1]['iteration'])))

    return os.path.join(experiment_folder, str(experiment_metrics[-1]['iteration']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-folderPath','-path',required=True, 
                        help='Path to folder containing metrics.json.')

    args = parser.parse_args()

    saveTrainValPlot(args.folderPath)
