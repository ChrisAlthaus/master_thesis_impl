from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import datetime
import os
import time

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-predictions', '-preds',  help='Path to a prediction json file in coco format.')
parser.add_argument('-gt_annotations', '-gt_ann', help='Path to gt annotation json file in coco format.')
parser.add_argument('-outputdir')
#Calculate predictions on-the-fly
parser.add_argument('-model_cp', help='Path to the model checkpoint (no prediction file needed).')
parser.add_argument('-imagedir', help='Path to the image dir which should be predicted.')


args = parser.parse_args()

def latestdir(dir):
    diritems = [os.path.join(dir, d) for d in os.listdir(dir)]
    all_subdirs = [d for d in diritems if os.path.isdir(d)]
    return max(all_subdirs, key=os.path.getmtime)

predictionfile = ''
# ---------------------------------------- PREDICTION ----------------------------------------
#When no prediction file is specified, previously calculate predictions 
if args.model_cp:
    print("MASK-RCNN PREDICTION:")
    gpu_cmd = '/home/althausc/master_thesis_impl/scripts/singularity/ubuntu_srun_G1d4-1.sh'
    topk = 100 #20
    score_tresh = 0.8 #0.7
    styletransfered = True

    cmd = "{} python3.6 /home/althausc/master_thesis_impl/scripts/detectron2/MaskRCNN_prediction.py -model_cp {} -imgdir {} -topk {} -score_tresh {} -target train {} -visrandom"\
                                                    .format(gpu_cmd, args.model_cp, args.imagedir, topk, score_tresh, '-styletransfered' if styletransfered else ' ')
    print(cmd)
    os.system(cmd)
    time.sleep(10)

    out_dir = '/home/althausc/master_thesis_impl/detectron2/out/art_predictions/train'
    predictionfile = os.path.join(latestdir(out_dir), 'maskrcnn_predictions.json')
else:
    assert args.predictions is not None
    predictionfile = args.predictions


# ---------------------------------------- EVALUATION -----------------------------------------------
outputdir = os.path.join(args.outputdir, datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
os.makedirs(outputdir)

with open(os.path.join(outputdir, 'config.txt'), 'a') as f:
    f.write("Prediction Src Flie: %s"%predictionfile + os.linesep)
    f.write("Groundtruth Annotations: %s"%args.gt_annotations + os.linesep)
    if args.model_cp:
        f.write("Model Checkpoint: %s"%args.model_cp + os.linesep)
        f.write("Image Directory for Prediction: %s"%args.imagedir + os.linesep)

print("Prediction-File: ", predictionfile)
print("Groundtruth Annotation-File: ", args.gt_annotations)

annType = ['segm','bbox','keypoints']
#annType = annType[1]      #specify type here
#prefix = 'person_keypoints' if annType=='keypoints' else 'instances'

cocoGt=COCO(args.gt_annotations)
cocoDt=cocoGt.loadRes(predictionfile)

imgIds=sorted(cocoGt.getImgIds())

evalstr = ''
evalstr = evalstr + "----------- EVALUATION FOR KEYPOINTS -------------" + os.linesep
# running evaluation
#Path of Cocoeval: .local/lib64/python3.6/site-packages/pycocotools/cocoeval.py
cocoEval = COCOeval(cocoGt,cocoDt,'keypoints')
cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
evalstr = evalstr + cocoEval.evalresults + os.linesep

evalstr = evalstr + "-------------- EVALUATION FOR BBOX ----------------" + os.linesep
# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
evalstr = evalstr + cocoEval.evalresults + os.linesep

with open(os.path.join(outputdir, 'evaluations.txt'), 'w') as f:
    f.write(evalstr)
print("Wrote evaluation summary to: ", outputdir)