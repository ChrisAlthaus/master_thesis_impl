import argparse
import os

from PIL import Image

#Script gets images from directory which cannot be opened

parser = argparse.ArgumentParser()
parser.add_argument('-imagedir', help='Directory with the new images.')
args = parser.parse_args()

img_files = [f for f in os.listdir(args.imagedir) if os.path.isfile(os.path.join(args.imagedir, f))]

corrupted = []

for i,img_name in enumerate(img_files):
    try:
        im = Image.open(os.path.join(args.imagedir,img_name))
    except:
        print("Failed to open image: ",img_name)
        corrupted.append(img_name)

print("Corrupted images: ")
print(*corrupted, sep=' ')
