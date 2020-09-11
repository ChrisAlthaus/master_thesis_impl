import h5py

import argparse
import os



parser = argparse.ArgumentParser()
parser.add_argument('-file', help='Path to a json file.')

args = parser.parse_args()

f = h5py.File(args.file)

print("root items:", list(f))
for item in list(f):
    print(item + ": " + str(f[item].shape))
    print(f[item][:10])
    print()