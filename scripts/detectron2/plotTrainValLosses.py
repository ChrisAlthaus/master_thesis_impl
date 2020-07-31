import json
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter

def saveTrainValPlot(experiment_folder):
    def load_json_arr(json_path):
        lines = []
        with open(json_path, 'r') as f:
            for line in f:
                lines.append(json.loads(line))
        return lines

    experiment_metrics = load_json_arr(experiment_folder + '/metrics.json')
    
    #Expand figure width for every 100 datapoints
    figsize = [6.4, 4.8]    #defaults
    figsize[1] = figsize[1] * max(1,len(experiment_metrics)/100)

    plt.figure(figsize=tuple(figsize))
    x_total = [x['iteration'] for x in experiment_metrics]
    y_total = [x['total_loss'] for x in experiment_metrics]
    plt.plot(x_total, y_total, label='total_loss')
    
    x_val = [x['iteration'] for x in experiment_metrics if 'validation_loss' in x]
    y_val = [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x]
    plt.plot(x_val, y_val, label='validation_loss')

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