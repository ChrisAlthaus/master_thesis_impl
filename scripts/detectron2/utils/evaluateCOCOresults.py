from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import datetime
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-predictions', '-preds',  help='Path to a prediction json file in coco format.')
parser.add_argument('-gt_annotations', '-gt_ann', help='Path to gt annotation json file in coco format.')
parser.add_argument('-outputdir')

args = parser.parse_args()

outputdir = os.path.join(args.outputdir, datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
os.makedirs(outputdir)

with open(os.path.join(outputdir, 'config.txt'), 'a') as f:
    f.write("Prediction Src Flie: %s"%args.predictions + os.linesep)
    f.write("Groundtruth Annotations: %s"%args.gt_annotations + os.linesep)

annType = ['segm','bbox','keypoints']
#annType = annType[1]      #specify type here
#prefix = 'person_keypoints' if annType=='keypoints' else 'instances'

cocoGt=COCO(args.gt_annotations)
cocoDt=cocoGt.loadRes(args.predictions)

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