import h5py

import argparse
import os



parser = argparse.ArgumentParser()
parser.add_argument('-file', help='Path to a json file.')
parser.add_argument('-firstn',type=int)

args = parser.parse_args()

f = h5py.File(args.file, 'r')

if args.firstn is not None:
    print(str(f)[:args.firstn])
    exit(1)
    
print("root items:", list(f))
for item in list(f):
    print(item + ": " + str(f[item].shape))
    print(f[item][:10])
    print()
exit(1)

#root items: ['active_object_mask', 'attributes', 'boxes_1024', 'boxes_512', 'img_to_first_box', 'img_to_first_rel', 'img_to_last_box', 'img_to_last_rel', 'labels', 'predicates', 'relationships', 'split']

print(f['img_to_first_box'][1080:1090])
for i in range(len(f['img_to_first_box'])-1):
    if f['img_to_first_box'][i+1] - f['img_to_first_box'][i] == 1:
        print(i)