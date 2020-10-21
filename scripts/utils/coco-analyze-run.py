import os

_MODES = ['Mask-RCNN', 'PoseFix']
_MODE = _MODES[0]

gtann = '/home/althausc/nfs/data/coco_17_medium/annotations_styletransfer/person_keypoints_val2017_stAPI.json'
predictions = '/home/althausc/master_thesis_impl/detectron2/out/art_predictions/train/10-13_12-49-15/maskrcnn_predictions.json'
teamname = 'artinfer'
version = '1.0'

if _MODE == _MODES[0]:
    outputdir = '/home/althausc/master_thesis_impl/results/posedetection/maskrcnn'
else:
    outputdir = '/home/althausc/master_thesis_impl/results/posedetection/posefix'

outputdir = os.path.join(outputdir, os.path.basename(os.path.dirname(predictions)))
if not os.path.exists(outputdir):
     os.makedirs(outputdir)
     print("Successfully created output directory: ", outputdir)

logfile = os.path.join(outputdir, 'log.txt')
os.chdir('/home/althausc/master_thesis_impl/coco-analyze/')
cmd = 'python3.6 run_analysis.py {} {} {} {} {} &> {}'\
             .format(gtann, predictions, outputdir, teamname, version, logfile)
print(cmd)
os.system(cmd)
