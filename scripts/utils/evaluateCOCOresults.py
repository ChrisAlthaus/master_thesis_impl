from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-predictions', '-preds',  help='Path to a prediction json file in coco format.')
parser.add_argument('-gt_annotations', '-gt_ann', help='Path to gt annotation json file in coco format.')

args = parser.parse_args()


annType = ['segm','bbox','keypoints']
#annType = annType[1]      #specify type here
#prefix = 'person_keypoints' if annType=='keypoints' else 'instances'

cocoGt=COCO(args.gt_annotations)
cocoDt=cocoGt.loadRes(args.predictions)

imgIds=sorted(cocoGt.getImgIds())

print("----------- EVALUATION FOR KEYPOINTS -------------")
# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,'keypoints')
cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
print("--------------------------------------------------")

print("----------- EVALUATION FOR BBOX -------------")
# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
print("--------------------------------------------------")