from itertools import product
import json
import os

data = []
x = [i for i in product(range(2), repeat=4)]

for k, bitvector in enumerate(x):
    data.append({'image_id':k , 'gpd': bitvector, 'mask':'1111', 'score':1})

print("Data: ", data)

outputdir = '/home/althausc/master_thesis_impl/posedescriptors/out/eval/bitseq4'
with open(os.path.join(outputdir, 'gpdsminus.json'), 'w') as f:
    print("Writing to folder: ",outputdir)
    json.dump(data, f, indent=2)