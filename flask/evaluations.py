import os
import json
from collections import defaultdict
import numpy as np

#Generate groundtruth annotation set
filepaths1 = []
annfolder = '/home/althausc/master_thesis_impl/flask/results/testresults'
humanposefolders = ['JJo_reduced', 'Jc_rel', 'JcJLdLLa_reduced', 'JLd_all_direct', 'random'] 

for descriptortype in humanposefolders:
    descriptordir = os.path.join(annfolder, descriptortype)
    onlyfiles = [os.path.join(descriptordir, f) for f in os.listdir(descriptordir) if os.path.isfile(os.path.join(descriptordir, f))]
    filepaths1.extend(onlyfiles)

backgrounds = ['art_history', 'computer_science', 'others']
groundtruth = {'positives': defaultdict(list), 'negatives': defaultdict(list)}
groundtruth_b = {}
for b in backgrounds:
    groundtruth_b.update({b: {'positives': defaultdict(list), 'negatives': defaultdict(list)}})

id_gt1 = defaultdict(set)
id_annlists1 = defaultdict(list)
id_annimgpaths1 = defaultdict(set)
id_retimgpaths1 = defaultdict(list)

for fpath in filepaths1:
    with open(fpath) as f:
        annitem = json.load(f)
    qname = os.path.splitext(os.path.basename(annitem['query']))[0]
    b = annitem['background']

    posfilenames = [annitem['retrievalfiles'][str(int(index)-1)]['filename'] for index in annitem['annotations']]
    negfilenames = [annitem['retrievalfiles'][str(int(index)-1)]['filename'] for index in annitem['negativs']]
    
    groundtruth_b[b]['positives'][qname].extend(posfilenames)
    groundtruth_b[b]['negatives'][qname].extend(negfilenames)
    groundtruth['positives'][qname].extend(posfilenames)
    groundtruth['negatives'][qname].extend(negfilenames)



    #id_gt1[qrid].update(annitem['annotations']) 
    #
    #id_annlists1[qrid].append(annitem['annotations'])
    #for sannid in annitem['annotations']:
    #    id_annimgpaths1[qrid].add(annitem['retrievalfiles'][sannid]['filename'])
    #id_retimgpaths1[qrid].append(sorted(annitem['retrievalfiles'].items(), key= lambda x: int(x[0])))


searchfolders = {'L1': '/home/althausc/master_thesis_impl/flask/results/testresults'}
humanposefolders = ['JJo_reduced', 'Jc_rel', 'JcJLdLLa_reduced', 'JLd_all_direct', 'random'] 

results_precision = {}
for h in humanposefolders:
    results_precision[h] = {}

for metric, basefolder in searchfolders.items():
    
    for descriptortype in humanposefolders:
        results = defaultdict(list)
        descriptordir = os.path.join(basefolder, descriptortype)
        onlyfiles = [os.path.join(descriptordir, f) for f in os.listdir(descriptordir) if os.path.isfile(os.path.join(descriptordir, f))]
        print(onlyfiles, os.path.join(basefolder, descriptortype))
        for fpath in onlyfiles:
            with open(fpath) as f:
                sranking = json.load(f)
            qname = os.path.splitext(os.path.basename(sranking['query']))[0]
            print(fpath)
            retrievalfiles = sorted(list(sranking['retrievalfiles'].items()), key= lambda x: int(x[0]))
            positives = [] # saves for each rank if image is positive (1) or not (0) 
            for r, item in retrievalfiles:
                rfilename = item['filename']
                if rfilename in groundtruth['positives'][qname]:
                    positives.append(1)
                else:
                    positives.append(0)
            print(positives)
            results[qname].append(positives)

        print(list(results))
        p5s = []
        p10s = []
        p20s = []
        for qname, positives in results.items():
            print(positives)
            n5 = np.sum(positives[:5])
            n10 = np.sum(positives[:10])
            n20 = np.sum(positives[:20])

            p5s.append(n5/5)
            p10s.append(n10/10)
            p20s.append(n20/20)
        print(p5s)
        print(p10s)
        print(p20s)
        print()
        p5 = np.mean(p5s)
        p10 = np.mean(p10s)
        p20 = np.mean(p20s)
        results_precision[descriptortype][metric] = {"P@5": p5, "P@10": p10, "P@20": p20}

print(results_precision)
    



"""
#Generate comparison annotation set
filepaths2 = []
descriptor = 'JcJLdLLa_reduced'
for path, subdirs, files in os.walk('results/%s'%descriptor):
	for name in files:
		filepath = os.path.join(path, name)
		if 'feedback' in filepath:
			continue
        filepaths2.append(filepath)

id_retimgpaths2 = defaultdict(list)

for fpath in filepaths1:
    with open(fpath) as f:
		annitem = json.load(f)
    qrid = annitem['id']
    id_retimgpaths2[qrid].append(sorted(annitem['retrievalfiles'].items(), key= lambda x: int(x[0])))


#Calculate evaluation scores
id_retimgpaths = None
evaluationtype = "goundtruth" #"comparison"
if evaluationtype == "groundtruth":
    id_retimgpaths = id_retimgpaths1
else:
    id_retimgpaths = id_retimgpaths2

ks =[5,10,20]
id_mAps = defaultdict(dict)
mode = "rankedprecision" #"normalprecision"

id_mAps['metadata'].update({'evaluation': evaluationtype, 'mode': mode})
if evaluationtype == "comparison":
    id_mAps['metadata'].update({'descriptor': descriptor})


for qrid, rimgpathslist in id_retimgpaths.items():
    gt_imgpaths = id_annimgpaths1[qrid]
    for k in ks:
        APks_id = []
        #all annotations for one query-result prediction
        for rimgpaths in rimgpathslist:
            relevant = []
            rnum = 0
            #one single (ranked) query-results annotation
            for r, ritem in enumerate(rimgpaths, start=1):
                if ritem[1]['filename'] in gt_imgpaths:
                    assert int(ritem[0]) == r
                    if mode=="rankedprecision":
                        rnum += 1
                        relevant.append(rnum/r)
                    else:
                        relevant.append(1)

            if mode=="rankedprecision":
                precision = np.mean(relevant)
            else:
                precision = np.sum(relevant)
            APks_id.append(precision)
            
        mAPk_id = np.mean(APks_id)
        id_mAps[qrid].update('mAP@{}'.format(k) : mAPk_id)

print("Mode: ", mode)
print(id_mAps)

if evaluationtype == "comparison":
    filename = 'mAP_{}_{}_perid.json'.format(descriptor, mode)
    filename_total = 'mAP_{}_{}_perid_total.json'.format(descriptor, mode)
else:
    filename = 'mAP_groundtruth_{}_perid.json'.format(mode)
    filename_total = 'mAP_groundtruth_{}_perid_total.json'.format(mode)


with open(os.path.join('/home/althausc/master_thesis_impl/results/userstudy', filename), 'w') as f:
    json.dump(id_mAps, f, indent=2)

mApstotal = defaultdict(list)
for apdict in id_mAps.values():
    for k in ks:
        mApstotal['mAP@{}'.format(k)].append(apdict['mAP@{}'.format(k)])
for k in ks:
    mApstotal['mAP@{}'.format(k)] = np.mean(apdict['mAP@{}'.format(k)])

with open(os.path.join('/home/althausc/master_thesis_impl/results/userstudy', filename_total), 'w') as f:
    json.dump(mApstotal, f, indent=2)
"""