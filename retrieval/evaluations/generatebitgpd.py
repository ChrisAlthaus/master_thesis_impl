from itertools import product
import json
import os

data = []
x = [i for i in product(range(2), repeat=4)]
y = [i for i in product(range(2), repeat=4)]

for k, bitvectors in enumerate(zip(x,y)):
    data.append({'image_id':k , 'gpd': bitvectors[0], 'mask': ''.join(map(str,bitvectors[1])), 'score':1})

print("Data: ", data)

outputdir = '/home/althausc/master_thesis_impl/posedescriptors/out/eval/bitseq4'
with open(os.path.join(outputdir, 'gpdsminus.json'), 'w') as f:
    print("Writing to folder: ",outputdir)
    json.dump(data, f, indent=2)