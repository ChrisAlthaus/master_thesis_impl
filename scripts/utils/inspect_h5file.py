import h5py

import argparse
import os
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('-file', help='Path to a json file.')
parser.add_argument('-firstn',type=int)
parser.add_argument('-searchindex',type=int)


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

traininds = [k for k in f['split'] if k==0]
valinds = [k for k in f['split'] if k==2]
print("Number of train images: ",len(traininds))
print("Number of validation images: ",len(valinds))

exit(1)

if args.searchindex is not None:
    for i in reversed(range(2,10)):
        print('img_to_first_box: ', f['img_to_first_box'][args.searchindex-i])
        print('img_to_last_box: ', f['img_to_last_box'][args.searchindex-i])
        print('img_to_first_rel: ', f['img_to_first_rel'][args.searchindex-i])
        print('img_to_last_rel: ', f['img_to_last_rel'][args.searchindex-i])
        print("--------------------------------")
    print('img_to_first_box: ', f['img_to_first_box'][args.searchindex-1])
    print('img_to_last_box: ', f['img_to_last_box'][args.searchindex-1])
    print('img_to_first_rel: ', f['img_to_first_rel'][args.searchindex-1])
    print('img_to_last_rel: ', f['img_to_last_rel'][args.searchindex-1])
    print("--------------------------------")
    print('img_to_first_box: ', f['img_to_first_box'][args.searchindex])
    print('img_to_last_box: ', f['img_to_last_box'][args.searchindex])
    print('img_to_first_rel: ', f['img_to_first_rel'][args.searchindex])
    print('img_to_last_rel: ', f['img_to_last_rel'][args.searchindex])
    print("--------------------------------")
    print('img_to_first_box: ', f['img_to_first_box'][args.searchindex+1])
    print('img_to_last_box: ', f['img_to_last_box'][args.searchindex+1])
    print('img_to_first_rel: ', f['img_to_first_rel'][args.searchindex+1])
    print('img_to_last_rel: ', f['img_to_last_rel'][args.searchindex+1])
    exit(1)

for i in range(len(f['img_to_first_box'])):
   # if f['img_to_first_box'][i] == -1:
   #     print('box index: ',i)

    if f['img_to_first_box'][i] == -1 and f['img_to_first_rel'][i] == -1:
        print('img_to_first_box: ', f['img_to_first_box'][i])
        print('img_to_last_box: ', f['img_to_last_box'][i])
        print('img_to_first_rel: ', f['img_to_first_rel'][i])
        print('img_to_last_rel: ', f['img_to_last_rel'][i])
        print("--------------------------------")


print(type(f['split']))
splitinds=[]
for i in f['split']:
    if i not in splitinds:
        splitinds.append(i)
print(splitinds)
traincount = 0
testcount = 0
for i in f['split']:
    if i == 0:
        traincount = traincount + 1
    elif i == 2:
        testcount = testcount + 1
print(traincount, testcount)


exit(1)


print(f['img_to_first_box'][len(f['img_to_first_box'])-100:])
print(f['img_to_last_box'][len(f['img_to_last_box'])-100:])

print(len(f['split']),type( f['split'][()]))
print([f['split'][()] == 0][:20])
print("split 0s: ",np.count_nonzero(f['split'][()] == 0))
print("split 2s: ",np.count_nonzero(f['split'][()] == 2))

invalid = []
for x in f['img_to_first_box'][()]:
    if x < 0:
        invalid.append(x)
print(len(invalid))
#print(set(f['split']))

print("max rel: ",np.max(f['relationships'][()].flatten()))
print(f['relationships'].shape)

exit(1)

#root items: ['active_object_mask', 'attributes', 'boxes_1024', 'boxes_512', 'img_to_first_box', 'img_to_first_rel', 'img_to_last_box', 'img_to_last_rel', 'labels', 'predicates', 'relationships', 'split']

print(f['img_to_first_box'][1080:1090])
for i in range(len(f['img_to_first_box'])-1):
    if f['img_to_first_box'][i+1] - f['img_to_first_box'][i] == 1:
        print(i)