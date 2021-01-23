from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.builtin_meta import KEYPOINT_CONNECTION_RULES, COCO_PERSON_KEYPOINT_NAMES, COCO_PERSON_KEYPOINT_FLIP_MAP
from detectron2.data import detection_utils as utils
from detectron2.data.datasets.coco import load_coco_json
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.samplers import InferenceSampler
import itertools, copy
from detectron2.data.build import build_detection_test_loader
from detectron2.structures import Instances
from detectron2.structures import Boxes, BoxMode

from detectron2.data.datasets import register_coco_instances
import torch
import numpy as np

import argparse
import os

from pycocotools.coco import COCO
import cv2
from PIL import Image
import datetime
import json
import random

import sys
sys.path.insert(0, '/home/althausc/master_thesis_impl/scripts/utils') 
from statsfunctions import getwhiskersvalues
from utils import utilsart500k


parser = argparse.ArgumentParser()
parser.add_argument('-model_cp','-mc',required=True, 
                    help='Path to the model checkpoint file.')
parser.add_argument('-image_path','-img', 
                    help='Path to the image for which pose inference will be calculated.')
parser.add_argument('-image_folder','-imgdir', 
                    help='Path to a directory containing images only.')
parser.add_argument('-mode',
                    help='Wheather to load the image path(s) from given directory/file or from custom loading function.')
parser.add_argument('-topk', type=int, default=20,
                    help='Filter the predictions and take best k poses.')
parser.add_argument('-score_tresh', type=float, default=0.5,
                    help='Filter detected poses based on a score treshold.')
parser.add_argument('-visunfiltered', action='store_true',
                    help='Specify to visualize unfiltered/all predictions on images & save.')
parser.add_argument('-visfiltered', action='store_true',
                    help='Specify to visualize filtered/reduced predictions on images & save.')
parser.add_argument('-visfilteredrandom', action='store_true',
                    help='Specify to visualize filtered/reduced predictions on images & save.')
parser.add_argument('-visrandom','-validate', action='store_true',
                    help='Specify to randomy visualize k predictions.')
parser.add_argument('-vistresh', type=float, default=0.0,   
                    help='Specify a treshold for visualization.')   #not used
parser.add_argument('-styletransfered',action="store_true", 
                    help='Wheather to tranform image name to style-transform image id (used for style transfered images.')   
parser.add_argument('-target',
                    help='Whether to later use predictions for training other model or for querying.\
                         Output folder will then be different (train/single).')     
args = parser.parse_args()


if not os.path.isfile(args.model_cp) and args.model_cp != 'baseline':
    raise ValueError("Model file path not exists.")
if args.image_path is not None:
    if not os.path.isfile(args.image_path):
        raise ValueError("Image does not exists.")
if args.image_folder is not None:
    if not os.path.isdir(args.image_folder):
        raise ValueError("Image does not exists.")

if args.image_path is None and args.image_folder is None and args.mode == 'loadpaths':
    raise ValueError("Please specify an image or an image directory.")

if args.target not in ['train', 'query', 'eval']:
    raise ValueError("Please specify a valid prediction purpose.")
if args.score_tresh > 1 or args.score_tresh < 0:
    raise ValueError("Please specify a valid filter treshold.")

def main():
    # -------------------------------- GET IMAGE PATH(S) --------------------------------
    specialchars = 'ÆÐƎƏƐƔĲŊŒẞÞǷȜæðǝəɛɣĳŋœĸſßþƿȝĄƁÇĐƊĘĦĮƘŁØƠŞȘŢȚŦŲƯY̨Ƴąɓçđɗęħįƙłøơşșţțŧųưy̨ƴÁÀÂÄǍĂĀÃÅǺĄÆǼǢƁĆĊĈČÇĎḌĐƊÐÉÈĖÊËĚĔĒĘẸƎƏƐĠĜǦĞĢƔáàâäǎăāãåǻąæǽǣ'+\
                    'ɓćċĉčçďḍđɗðéèėêëěĕēęẹǝəɛġĝǧğģɣĤḤĦIÍÌİÎÏǏĬĪĨĮỊĲĴĶƘĹĻŁĽĿʼNŃN̈ŇÑŅŊÓÒÔÖǑŎŌÕŐỌØǾƠŒĥḥħıíìiîïǐĭīĩįịĳĵķƙĸĺļłľŀŉńn̈ňñņŋóòôöǒŏōõőọøǿơœŔŘŖŚŜŠŞȘ'+\
                    'ṢẞŤŢṬŦÞÚÙÛÜǓŬŪŨŰŮŲỤƯẂẀŴẄǷÝỲŶŸȲỸƳŹŻŽẒŕřŗſśŝšşșṣßťţṭŧþúùûüǔŭūũűůųụưẃẁŵẅƿýỳŷÿȳỹƴźżžẓ'
    specialchars = [c for c in specialchars]

    image_paths = []

    if args.mode == 'loadpaths' or args.target == 'query':
        if args.image_folder is not None:
            for path, subdirs, files in os.walk(args.image_folder): #os.walk(u'%s'%args.image_folder):
                for name in files:
                    imgpath = os.path.join(path, name)   
                    for c in specialchars:
                        if c in imgpath:
                            image_paths.append(imgpath) 
                            break
                        
        elif args.image_path is not None:
            image_paths = [args.image_path]
    elif args.mode == 'custompaths':
        image_paths = utilsart500k.get_paths_of_paintings()
    else:
        raise ValueError("Unvalid mode argument.")

    #image_paths = image_paths[:100]
    # -------------------------------- LOAD CFG & SET MODEL PARAMS -----------------------
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")) 
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml") 
    if args.model_cp != 'baseline': # no cfg.MODEL.WEIGHTS = default checkpoint provided by authors
        cfg.MODEL.WEIGHTS = args.model_cp 
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")

    #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.DEVICE= 'cuda' #'cpu'


    #Set necessary flags for freezing net
    cfg.MODEL.BACKBONE.FREEZE_AT = 5
    cfg.MODEL.BACKBONE.FREEZE_AT_ENABLED = True
    cfg.MODEL.BACKBONE.FREEZE_FROM = 0
    cfg.MODEL.BACKBONE.FREEZE_FROM_ENABLED = False
    cfg.MODEL.FPN.FREEZE = True

    # ------------------------------- INFER PREDICTIONS -------------------------------
    _TOPK = args.topk
    _SCORE_TRESH = args.score_tresh

    outputs = []
    outputs_raw = []
    imgpaths_abovetresh = []
    predictor = DefaultPredictor(cfg)
    print(cfg)
    print("SCORE TRESHOLD: ",_SCORE_TRESH)

    with torch.no_grad():
        print("START PREDICTION")
        #Providing prediction for single and multiple images
        batchsize = 1#10#2#10 #10-> OOM error
        #Percent of predictions not used
        notused = []
        #Number of images with no predictions
        nopreds = []

        if len(image_paths) < 10:
            batchsize = len(image_paths)
        for i in range(0, len(image_paths), batchsize):
            inputs = []
            imagepaths_batch = []
            for img_path in image_paths[i:i+batchsize]:
                #try:
                #stream = open(img_path.encode('utf-8'), "rb")
                #bytes = bytearray(stream.read())
                #print("0.1")
                #numpyarray = np.asarray(bytes, dtype=np.uint8)
                #print("0.2")
                #img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
                #    #img = cv2.imdecode(np.fromfile(u'{}'.format(img_path), dtype=np.uint8), cv2.IMREAD_UNCHANGED) #to account for special characters
                ##except cv2.error as e:
                ##    print("Image loading error")
                ##    print(e)
                ##    continue
                #img = cv2.imread(img_path)
                try:
                    img = np.array(Image.open(img_path.encode('utf-8'), 'r').convert('RGB'))
                    #img = cv2.imdecode(np.fromfile(img_path.encode('utf-8'), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                    #img = cv2.imread(img_path.encode('utf-8'))
                except Exception as e:
                    print(e)
                    continue

                if img is None:
                    print("None")
                    continue
                if len(img.shape) < 3:
                    print("Image dimensions not valid: ", img.shape)
                    continue
                if img.shape[2] > 3:
                    print("Reducing shape {} to dimension 3".format(img.shape))
                    img = img[:,:,:3]

                inputs.append(img)
                imagepaths_batch.append(img_path)
            if len(imagepaths_batch) == 0:
                continue
            #Prediction output:
            #For each image theres an instance-class, which format is:
            #   {'instances': Instances(num_instances=X, image_height=h, image_width=w, fields=[pred_boxes, scores, pred_classes, pred_keypoints])}
            preds = predictor(inputs)

            for img_path,pred in zip(imagepaths_batch, preds):
                if args.target == 'train':
                    #Remove image base directory, because the imageids should be relative
                    image_name = img_path.replace(args.image_folder, '').strip('/')
                else:
                    image_name = img_path

                if args.styletransfered: 
                    content_id = image_name.split('_')[0]
                    style_id = image_name.split('_')[1]
                    image_id = int("%s%s"%(content_id,style_id))
                else:
                    image_id = image_name #u'{}'.format(image_name)#.encode('utf-8') #allow string image id
                
                c = 0
                added = False
                for bbox, keypoints ,score in zip(pred["instances"].pred_boxes, pred["instances"].pred_keypoints, pred["instances"].scores.cpu().numpy()):
                    if score.astype("float") >= _SCORE_TRESH:
                        bbox = BoxMode.convert(bbox.tolist(), BoxMode.XYXY_ABS, BoxMode.XYWH_ABS) #Since coco-evaluation works with XYWH format
                        outputs.append({'image_id': image_id, 'image_size': pred["instances"]._image_size,  "category_id": 1, "bbox": bbox, "keypoints":keypoints.flatten().tolist(), "score": score.astype("float")})
                        added = True
                    if c >= _TOPK:
                        break
                    c = c + 1
                if len(pred["instances"]) != 0:
                    notused.append(1-c/(len(pred["instances"])))
                if added is False:
                    nopreds.append(img_path)
                outputs_raw.append(pred)
                #print(pred["instances"].scores)
                #print("original num of predictions: ",image_id, len(pred["instances"].pred_boxes), c)
                #assert len(pred["instances"].pred_boxes) == c, print(len(pred["instances"].pred_boxes), c)
            if i%1000 == 0 and i!=0:
                print("Processed %d images."%(i+batchsize))

        print("PREDICTION FINISHED")
        print("Percentage of not used predictions (averaged): ", np.mean(notused))
        print("Number of images with no predictions: ", len(nopreds))
        print("Number of images with predictions: ", len(image_paths) - len(nopreds))

        if len(image_paths)-len(nopreds)>0:
            print("Mean number of poses per image: ", len(outputs)/(len(image_paths) - len(nopreds)))   
    
        if len(outputs) == 0:
            print("\nNo predictions made.")
            print("Raw outputs: ", outputs_raw)
            exit(1)

    # ------------------------------ SAVE PREDICTIONS ------------------------------
    output_dir = os.path.join('/home/althausc/master_thesis_impl/detectron2/out/art_predictions', args.target)
    output_dir = os.path.join(output_dir, datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        raise ValueError("Output directory %s already exists."%output_dir)
 
    #output format of keypoints: (x, y, v), v indicates visibility— v=0: not labeled (in which case x=y=0), v=1: labeled but not visible, and v=2: labeled and visible  
    with open(os.path.join(output_dir, "maskrcnn_predictions.json"), 'w') as f: #, encoding='utf8'
        json.dump(outputs, f, separators=(', ', ': ')) #, ensure_ascii=False)

    # ------------------------------- SAVE RUN CONFIG -------------------------------
    if args.target == 'train':
        #Statistics: number of poses per image
        numposes_per_image = []
        for itemgroup in get_combined_predictions(outputs):
            numposes_per_image.append(len(itemgroup['bboxes']))
        statisticsstr = getwhiskersvalues(numposes_per_image)

        #Writing config to file
        with open(os.path.join(output_dir, 'config.txt'), 'a') as f:
            f.write("Src Image Folder: %s"%(args.image_folder if args.image_folder is not None else args.image_path) + os.linesep)
            f.write("Model Checkpoint: %s"%args.model_cp + os.linesep)
            f.write("Topk: %d"%args.topk + os.linesep)
            f.write("Score Treshold: %f"%args.score_tresh + os.linesep)
            f.write("Number of images: %d"%len(image_paths) + os.linesep)
            f.write("Percentage of not used predictions (averaged): %f"%np.mean(notused) + os.linesep)
            f.write("Number of images with no predictions: %d"%len(nopreds) + os.linesep)
            f.write("Number of images with predictions: %d"%(len(image_paths) - len(nopreds)) + os.linesep)
            f.write("Statistics of poses/image: \n%s"%statisticsstr + os.linesep)

    elif args.target == 'eval':
        #Writing config to file
        with open(os.path.join(output_dir, 'config.txt'), 'a') as f:
            f.write("Src Image Folder: %s"%(args.image_folder if args.image_folder is not None else args.image_path) + os.linesep)
            f.write("Model Checkpoint: %s"%args.model_cp + os.linesep)
            f.write("Topk: %d"%args.topk + os.linesep)
            f.write("Score Treshold: %f"%args.score_tresh + os.linesep)
            f.write("Number of images: %d"%len(image_paths) + os.linesep)
            
    # ------------------------- VISUALIZE PREDICTIONS ONTO SOME IMAGES FOR VALIDATION ---------------------- 
    
    MetadataCatalog.get("my_dataset_val").set(keypoint_names=COCO_PERSON_KEYPOINT_NAMES,
                                              keypoint_flip_map=COCO_PERSON_KEYPOINT_FLIP_MAP,
                                              keypoint_connection_rules=KEYPOINT_CONNECTION_RULES) 

    
    visdir = os.path.join(output_dir, '.visimages')
    if not os.path.exists(visdir):
        os.makedirs(visdir)
    visability_means = []

    if args.visunfiltered:
        #Draw all predictions of the model
        print("Visualize the predictions onto the original image(s) ...")
        
        print("Draw all/unfiltered predictions...")
        #Draw unfiltered predictions
        for img_path, pred_out in zip(image_paths, outputs_raw):
            visualize_and_save(img_path, visdir, pred_out, args, 'all')

            #Statistics
            for kpt_list in pred_out["instances"].pred_keypoints.cpu():
                kpt_list = kpt_list.numpy()
                visability_means.append(np.sum(kpt_list[:,2])/len(kpt_list))
        print("Draw all/unfiltered predictions done.")

    if args.visfiltered:
         #Draw only filtered predictions
        print("Draw topk + treshold predictions...")
        for preds in get_combined_predictions(outputs):
            visualize_and_save(preds['imagepath'], visdir, preds, args, 'treshtopk')
        print("Draw topk + treshold predictions done.")

        #Statistics
        for pred in outputs:
            kpt_list = pred['keypoints']
            visability_means.append(np.sum(kpt_list[2::3])/len(kpt_list))

    if args.visfilteredrandom:
         #Draw only filtered predictions
        preds_comb = get_combined_predictions(outputs)
        print("Draw topk + treshold predictions...")
        for i in range(100):
            pred = random.choice(preds_comb) #debug, print pred first
            visualize_and_save(pred['imagepath'], visdir, pred, args, 'treshtopk')
        print("Draw topk + treshold predictions done.")

        #Statistics
        for pred in outputs:
            kpt_list = pred['keypoints']
            visability_means.append(np.sum(kpt_list[2::3])/len(kpt_list))

    if args.visrandom:
        print("Random visualization for validation purposes ...")
        preds_comb = get_combined_predictions(outputs)
        for i in range(100):
            k = random.choice(range(len(image_paths)))
            img_path = image_paths[k]
            pred_out = outputs_raw[k]

            #Draw unfiltered predictions
            visualize_and_save(img_path, visdir, pred_out, args, 'all')
            #Draw topk predictions
            if any(x['imagepath'] == img_path for x in preds_comb):
                pred_searched = next(item for item in preds_comb if item["imagepath"] == img_path)
                visualize_and_save(img_path, visdir, pred_searched, args, 'treshtopk')

            #Statistics
            for kpt_list in pred_out["instances"].pred_keypoints.cpu():
                kpt_list = kpt_list.numpy()
                visability_means.append(np.sum(kpt_list[2::3])/len(kpt_list))
        print("Random visualization done.")
        
    print(visability_means)
    visstatstr = getwhiskersvalues(visability_means)
    #Writing config to file
    with open(os.path.join(output_dir, '.visstats.txt'), 'a') as f:
        f.write("Visualization mode: ")
        if args.visunfiltered:
            f.write("all unfiltered" + os.linesep)
        if args.visfiltered:
            f.write("all filtered" + os.linesep)
        if args.visrandom:
            f.write("random unfiltered" + os.linesep)
        f.write("Visability means stats: \n%s"%visstatstr + os.linesep)

   

    print("Output directory: ",output_dir)


def visualize_and_save(img_path, output_dir, preds, args, mode):
    #try:
    #    print("Loading: ",img_path.encode('utf-8'))
    #    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED) #to account for special characters
    #except cv2.error as e:
    #    print("Image loading error: ", e)
    #    return

    try:
        #img_path = os.path.join(args.image_folder, img_path)
        print("Loading: ",img_path.encode('utf-8'))
        img = np.array(Image.open(img_path.encode('utf-8'), 'r').convert('RGB'))
    except Exception as e:
        print(e)
        return

    if img is None:
        print("Warning: Image is none.")
        return

    if mode == 'all':
        #Draw unfiltered predictions, format = raw model output
        v = Visualizer(img[:, :, ::-1],MetadataCatalog.get("my_dataset_val"), scale=1.2)
        out = v.draw_instance_predictions(preds["instances"].to("cpu"), args.vistresh)
        img_name = os.path.basename(img_path)
        if out == None:
            print("Warning: Image is none.")
        Image.fromarray(out.get_image()[:, :, ::-1]).save(os.path.join(output_dir, img_name))

    elif mode == 'treshtopk':
        #Draw topk predictions
        v = Visualizer(img[:, :, ::-1],MetadataCatalog.get("my_dataset_val"), scale=1.2)
    
        obj = Instances(image_size=preds["imagesize"])
        obj.set('scores', torch.Tensor(preds["scores"]))
        obj.set('pred_boxes', Boxes(torch.Tensor(preds["bboxes"])))
        obj.set('pred_keypoints', torch.Tensor(preds["keypoints"]))
        
        out = v.draw_instance_predictions(obj, args.vistresh)
        basenames = os.path.splitext(os.path.basename(img_path))
        img_name = os.path.join("%s_topk%s"%(basenames[0], basenames[1]))
        if out == None:
            print("Warning: Image is none.")
        print("Save visualization image to ", os.path.join(output_dir, img_name).encode('utf-8'))
        #cv2.imwrite(os.path.join(output_dir, img_name),out.get_image()[:, :, ::-1])
        #is_success, im_buf_arr = cv2.imencode(".jpg", out.get_image()[:, :, ::-1])
        #im_buf_arr.tofile(os.path.join(output_dir, img_name))
        print(out.get_image()[:, :, ::-1].shape)
        Image.fromarray(out.get_image()[:, :, ::-1]).save(os.path.join(output_dir, img_name))


def get_combined_predictions(singlepreds):
     #Zip single predictions (every annotation is one item) to composed prediction (like output of model)
    singlepreds = sorted(singlepreds, key=lambda x: x['image_id'])
    
    if args.image_folder is None:
        imgdir = os.path.dirname(args.image_path)
    else:
        imgdir = args.image_folder 

    grouped = {}
    for pred_entry in singlepreds:
        root, ext = os.path.splitext(pred_entry['image_id'])
        if not ext:
            imagepath = os.path.join(imgdir, "%s.jpg"%(pred_entry['image_id']))
        else:
            imagepath = os.path.join(imgdir, pred_entry['image_id'])
        
        if imagepath not in grouped:
            grouped[imagepath] = [pred_entry]
        else:
            grouped[imagepath].append(pred_entry)
   
    combined = []
    for img_path, pred_items in grouped.items():
        preds = {}
        preds['imagepath'] = img_path
        preds['imagesize'] = pred_items[0]['image_size']
        preds['keypoints'] = [ [p['keypoints'][i:i+3] for i in range(0, len(p['keypoints']), 3)] for p in pred_items]
        preds['scores'] = [ p['score'] for p in pred_items]
        preds['bboxes'] = [ p['bbox'] for p in pred_items]
        combined.append(preds)
    return combined

if __name__=="__main__":
    main()
