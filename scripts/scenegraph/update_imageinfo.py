import argparse
import os

import json
from PIL import Image

#Update VG image info (image_data.json) with width & height values of modified gt images
#Reason: Style Transfer usually adds or removes a pair of pixels of the input image

parser = argparse.ArgumentParser()
parser.add_argument('-file', help='Path to the json file containing image info.')
#e.g. sample entry: {'width': 800, 'url': 'https://cs.stanford.edu/people/rak248/VG_100K_2/1.jpg', 'height': 600, 'image_id': 1, 'coco_id': None, 'flickr_id': None, 'anti_prop': 0.0}
parser.add_argument('-imagedir', help='Directory with the new images.')
parser.add_argument('-outputdir', required=True)

args = parser.parse_args()

json_data = None
with open(args.file, "r") as f:
    json_data = json.load(f)

json_out = []
print("num images input: ", len(json_data))
img_files = [f for f in os.listdir(args.imagedir) if os.path.isfile(os.path.join(args.imagedir, f))]
img_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
print(img_files[len(img_files)-10:])

print("Number of images in directory: ",len(img_files))
print("Number of images in image info file: ",len(json_data))
corrupted = []
notfound = []

for i,img_name in enumerate(img_files):
    try:
        im = Image.open(os.path.join(args.imagedir,img_name))
        width, height = im.size
        img_id = int(os.path.splitext(img_name)[0])
        num_before = len(json_out)
        found = False
        for elem in json_data:
            if elem['image_id'] == img_id:
                elem['width'] = width
                elem['height'] = height
                found = True
                break
        if not found:
            print("Image id not found: ",img_id)
            notfound.append(img_name)
    except:
        print("Failed to open image: ",img_name)
        corrupted.append(img_name)

    if i%1000 == 0:
        print("Processed %d images."%i)

#Corrupted Images:  []
#Not found images in input info file: ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg', '2316942.jpg', '2417331.jpg']
#Number of corrupted & not found images: 6
print("Corrupted Images: ",corrupted)
print("Not found images in input info file:", notfound)
print("Number of corrupted & not found images:", len(corrupted) + len(notfound))
with open(os.path.join(args.outputdir, 'image_data_updated.json'), 'w') as f:
    print("Writing to file: ",os.path.join(args.outputdir, 'image_data_updated.json'))
    json.dump(json_data, f)

