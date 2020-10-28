import argparse
import os

from PIL import Image

#Compares the sizes of equal named images from two directories
parser = argparse.ArgumentParser()
parser.add_argument('-imgdir1')
parser.add_argument('-imgdir2')

args = parser.parse_args()

img_files1 = [f for f in os.listdir(args.imgdir1) if os.path.isfile(os.path.join(args.imgdir1, f))]
img_files1.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

img_files2 = [f for f in os.listdir(args.imgdir2) if os.path.isfile(os.path.join(args.imgdir2, f))]
img_files2.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

for imgfile1 in img_files1:
    for imgfile2 in img_files2:
        if imgfile1 == imgfile2:
            im1size = Image.open(os.path.join(args.imgdir1,imgfile1)).size
            im2size = Image.open(os.path.join(args.imgdir2,imgfile2)).size
            if im1size != im2size:
                print(imgfile1,im1size,im2size)
            break


