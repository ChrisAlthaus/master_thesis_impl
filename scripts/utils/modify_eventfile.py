import sys
from pathlib import Path
import os
import argparse
# Use this if you want to avoid using the GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from tensorflow.core.util.event_pb2 import Event
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import copy 

def event_swapfields(input_path, output_path, swap_tags):
    inv_map = {v: k for k, v in swap_tags.items()}
    # Make a record writer
    with tf.io.TFRecordWriter(str(output_path)) as writer:
        # Iterate event records
        for rec in tf.data.TFRecordDataset([str(input_path)]):
            # Read event
            ev = Event()
            ev.MergeFromString(copy.deepcopy(rec.numpy()))
            # Check if it is a summary
            if ev.summary:
                for v in ev.summary.value:
                    if v.tag in swap_tags:
                        print("{} -> {}".format(v.tag, swap_tags[v.tag]))
                        v.tag = swap_tags[v.tag]
                    else:
                        if v.tag in inv_map:
                            print("{} -> {}".format(v.tag, inv_map[v.tag]))
                            v.tag = inv_map[v.tag]
            writer.write(ev.SerializeToString())

    print_eventfile(input_path)
    print("-"*20)
    print_eventfile(output_path)

def print_eventfile(inputpath):
    event_acc = EventAccumulator(os.path.dirname(inputpath))
    event_acc.Reload()
    # Show all tags in the log file
    print(event_acc.Tags())
    # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
    #w_times, step_nums, vals = zip(*event_acc.Scalars('Accuracy'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-eventfile',required=True)
    parser.add_argument('-mode',required=True)
    args = parser.parse_args()

    if args.mode == 'swap-scenegraph-fasterrcnn-evaluations':
        eventdir = os.path.dirname(args.eventfile)
        eventname = os.path.split(args.eventfile)[1]
        outputdir = os.path.join(eventdir, 'eventfile-swapped')

        print(eventname)
        if not os.path.exists(os.path.join(eventdir, 'eventfile-swapped')):
            os.makedirs(outputdir)
        outputfile = os.path.join(outputdir, eventname+'-renamed' )

        swap = { 'bbox/AP': 'bbox/AP(2)' , 
                        'bbox/AP50': 'bbox/AP50(2)' ,
                        'bbox/AP75': 'bbox/AP75(2)' ,
                        'bbox/APl': 'bbox/APl(2)' ,
                        'bbox/APm': 'bbox/APm(2)' ,
                        'bbox/APs': 'bbox/APs(2)' ,
                        'R@100': 'R@100(2)' 
        }
        event_swapfields(args.eventfile, outputfile, swap)
        print("Output directory: ", outputdir)

     if args.mode == 'swap-scenegraph-reltrain-evaluations':
        eventdir = os.path.dirname(args.eventfile)
        eventname = os.path.split(args.eventfile)[1]
        outputdir = os.path.join(eventdir, 'eventfile-swapped')

        print(eventname)
        if not os.path.exists(os.path.join(eventdir, 'eventfile-swapped')):
            os.makedirs(outputdir)
        outputfile = os.path.join(outputdir, eventname+'-renamed' )

        swap = { 'recall/R@20': 'recall(2)/R@20' ,  
                'recall/R@50': 'recall(2)/R@50' , 
                'recall/R@100': 'recall(2)/R@100' , 
                'recall_mean/R@20': 'recall_mean(2)/R@20' , 
                'recall_mean/R@50': 'recall_mean(2)/R@50' , 
                'recall_mean/R@100': 'recall_mean(2)/R@100' , 
                'recall_ng_mean/R@20': 'recall_ng_mean(2)/R@20' , 
                'recall_ng_mean/R@50': 'recall_ng_mean(2)/R@50' , 
                'recall_ng_mean/R@100': 'recall_ng_mean(2)/R@100' ,  
                'recall_ng_zero/R@20': 'recall_ng_zero(2)/R@20' , 
                'recall_ng_zero/R@50': 'recall_ng_zero(2)/R@50' , 
                'recall_ng_zero/R@100': 'recall_ng_zero(2)/R@100' , 
                'recall_nogc/R@20': 'recall_nogc(2)/R@20' , 
                'recall_nogc/R@50': 'recall_nogc(2)/R@50' , 
                'recall_nogc/R@100': 'recall_nogc(2)/R100' ,
                'recall_zero/R@20': 'recall_zero(2)/R@20' , 
                'recall_zero/R@50': 'recall_zero(2)/R@50' , 
                'recall_zero/R@100': 'recall_zero(2)/R@100' ,     
        }
        event_swapfields(args.eventfile, outputfile, swap)  #TODO:testing
        print("Output directory: ", outputdir)


    print('Done')

