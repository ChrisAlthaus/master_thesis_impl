
import json
from PIL import Image
import cv2
import numpy as np
import os



predictionfile = '/home/althausc/master_thesis_impl/detectron2/out/art_predictions/train/01-22_13-09-01/maskrcnn_predictions.json'

with open (predictionfile, "r") as f:
        json_data = json.load(f)
 
json_data = sorted(json_data, key=lambda k: k['image_id'])
for item in json_data:
    img_path = item['image_id']
    #try:
    #    Image.open(img_path.encode('utf-8'), 'r')
    #except Exception as e:
    #    print(e)

    #img = cv2.imdecode(np.fromfile(img_path.encode('utf-8'), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    image_name = os.path.relpath(img_path, os.path.dirname(os.path.normpath('/nfs/data/iart/art500k/img'))).replace('../','')
    print(image_name.encode('utf-8'))
    image_name= img_path.replace('/nfs/data/iart/art500k/img', '')
    print(image_name.encode('utf-8'))
    print(os.path.join('/nfs/data/iart/art500k/img', image_name.strip('/')))