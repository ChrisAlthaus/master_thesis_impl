import json
import os

jsonAnnotationFile = '/nfs/data/coco_17/annotations/person_keypoints_val2017.json'
searchid = 163682

with open (jsonAnnotationFile, "r") as f:
    json_data = json.load(f)

images = json_data['images']
annotations = json_data['annotations']
annotations_updated = {'images':[], 'annotations':[]}   

for item in images:
    if item['id'] == searchid:
        annotations_updated['images'].append(item)


for item in annotations:
    if item['image_id'] == searchid:
        annotations_updated['annotations'].append(item)

print("Updated \n: ", annotations_updated)
annotations_updated['categories'] = json_data['categories']

output_dir = '/home/althausc/master_thesis_impl/detectron2/out/art_predictions/query/11-13_10-46-14'
with open(os.path.join(output_dir, 'annsingle.json'), 'w') as f:
    print("Writing to folder: ",output_dir)
    json.dump(annotations_updated, f)