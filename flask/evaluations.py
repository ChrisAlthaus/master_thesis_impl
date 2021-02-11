import os
import json
from collections import defaultdict
import numpy as np
import math

def main():
    # --------------------------------------------------------- HUMAN POSES --------------------------------------------------
    print("HUMAN POSES")
    #Generate groundtruth annotation set
    filepaths1 = []
    annfolder = '/home/althausc/master_thesis_impl/flask/results/testresults'
    humanposefolders = ['JJo_reduced', 'Jc_rel', 'JcJLdLLa_reduced', 'JLd_all_direct', 'random'] 

    for descriptortype in humanposefolders:
        descriptordir = os.path.join(annfolder, descriptortype)
        onlyfiles = [os.path.join(descriptordir, f) for f in os.listdir(descriptordir) if os.path.isfile(os.path.join(descriptordir, f))]
        filepaths1.extend(onlyfiles)

    backgrounds = ['art_history', 'computer_science', 'others']
    groundtruth = {'positives': defaultdict(list), 'negatives': defaultdict(list), 'dontknow': defaultdict(list)}
    groundtruth_b = {}
    for b in backgrounds:
        groundtruth_b.update({b: {'positives': defaultdict(list), 'negatives': defaultdict(list), 'dontknow': defaultdict(list)}})

    print("Human Pose annotations:", filepaths1)
    print("Number of annotations", len(filepaths1))
    print()

    for fpath in filepaths1:
        with open(fpath) as f:
            annitem = json.load(f)
        qname = os.path.splitext(os.path.basename(annitem['query']))[0]
        b = annitem['background']

        posfilenames = [annitem['retrievalfiles'][str(int(index)-1)]['filename'] for index in annitem['annotations']]
        negfilenames = [annitem['retrievalfiles'][str(int(index)-1)]['filename'] for index in annitem['negativs']]

        dontknow_indices = [i for i in list(range(1,21)) if i not in annitem['annotations'] + annitem['negativs']]
        dknowfilenames = [annitem['retrievalfiles'][str(int(index)-1)]['filename'] for index in dontknow_indices]

        groundtruth_b[b]['positives'][qname].extend(posfilenames)
        groundtruth_b[b]['negatives'][qname].extend(negfilenames)
        groundtruth_b[b]['dontknow'][qname].extend(dknowfilenames)
        groundtruth['positives'][qname].extend(posfilenames)
        groundtruth['negatives'][qname].extend(negfilenames)
        groundtruth['dontknow'][qname].extend(dknowfilenames)

    stats = "Human Pose Groundtruth Statistics:" + '\n'
    for qname in groundtruth['positives'].keys():
        txt1 = "Number of relevant images for query {} = {}".format(qname, len(set(groundtruth['positives'][qname])))
        txt2 = "Number of non-relevant images for query {} = {}".format(qname, len(set(groundtruth['negatives'][qname])))
        print(txt1)
        print(txt2)
        stats += txt1 + '\n'
        stats += txt2 + '\n'

    print("-----------------------------------------------------")


    searchfolders = {'L1': '/home/althausc/master_thesis_impl/retrieval/out/userstudy/data-L1', 
                    'L2': '/home/althausc/master_thesis_impl/retrieval/out/userstudy/data-L2',
                    'COSSIM': '/home/althausc/master_thesis_impl/retrieval/out/userstudy/data-cossim'}
    humanposefolders = ['JJo_reduced', 'Jc_rel', 'JcJLdLLa_reduced', 'JLd_all_direct', 'random'] 

    results_precision = {}
    for h in humanposefolders:
        results_precision[h] = {}

    for metric, basefolder in searchfolders.items():

        for descriptortype in humanposefolders: #evaluate each descriptor type independent
            print('Calculation: {} {}'.format(metric, descriptortype))
            results = defaultdict(list) #query image -> array of 0/1 (1 indicate a positive)
            results_graded = defaultdict(list) #query image -> array of 0/1/2 (0:negatives, 1:don't know, 2: positives)
            metadatadir = os.path.join(basefolder, descriptortype, 'metadata')
            onlyfiles = [os.path.join(metadatadir, f) for f in os.listdir(metadatadir) if os.path.isfile(os.path.join(metadatadir, f))]
            print("Calculating metrics for {} files in directory {}.".format(len(onlyfiles), metadatadir))

            for fpath in onlyfiles:
                with open(fpath) as f:
                    sranking = json.load(f)

                qname = os.path.splitext(os.path.basename(sranking['querypath']))[0]
                #print(fpath)
                retrievalfiles = sorted(list(sranking['retrievaltopk'].items()), key= lambda x: int(x[0]))
                positives = [] # saves for each rank if image is positive (1) or not (0) 
                graded = [] # saves for each rank 0,1,2, (0 for negatives and not seen images, 1= dontknow, 2= positives)
                for r, item in retrievalfiles:
                    rfilename = item['filename']
                    if rfilename in groundtruth['positives'][qname]:
                        positives.append(1)
                        graded.append(2)
                    else:
                        positives.append(0)
                        if rfilename in groundtruth['negatives'][qname]:
                            graded.append(0)
                        elif rfilename in groundtruth['dontknow'][qname]:
                            graded.append(1)
                        else:
                            graded.append(0)
               
                results[qname].append(positives)
                results_graded[qname].append(graded)

            results_precision[descriptortype][metric] = getresults_queryimages(results, results_graded, groundtruth, descriptortype)


    print("Results:")
    for descriptor,metricresults in results_precision.items():
        print(descriptor)
        for metric, results in metricresults.items():
            print(metric, results)

    filename = 'humanpose-results.json'
    with open(os.path.join('/home/althausc/master_thesis_impl/results/userstudy', filename), 'w') as f:
        json.dump(results_precision, f, indent=2)

    #------------------------------------------------- SCENE GRAPHS --------------------------------------------------
    print("SCENE GRAPHS:")
    #Generate groundtruth annotation set
    filepaths2 = []
    annfolder = '/home/althausc/master_thesis_impl/flask/results/testresults'
    scenegraphfolders = ['scenegraphs', 'scenegraphs_random'] 

    for folder in scenegraphfolders:
        descriptordir = os.path.join(annfolder, folder)
        onlyfiles = [os.path.join(descriptordir, f) for f in os.listdir(descriptordir) if os.path.isfile(os.path.join(descriptordir, f))]
        filepaths2.extend(onlyfiles)

    print("Scene graph annotations", filepaths2)
    print("Number of annotations", len(filepaths2))
    print()

    backgrounds_sg = ['art_history', 'computer_science', 'others']
    groundtruth_sg = {'positives': defaultdict(list), 'negatives': defaultdict(list), 'dontknow': defaultdict(list)}
    groundtruth_b_sg = {}
    for b in backgrounds_sg:
        groundtruth_b_sg.update({b: {'positives': defaultdict(list), 'negatives': defaultdict(list), 'dontknow': defaultdict(list)}})


    print("Annotation files used for groundtruth set:", filepaths2)
    print()

    for fpath in filepaths2:
        with open(fpath) as f:
            annitem = json.load(f)
        qname = os.path.splitext(os.path.basename(annitem['query']))[0]
        b = annitem['background']

        posfilenames = [annitem['retrievalfiles'][str(int(index)-1)]['filename'] for index in annitem['annotations']]
        negfilenames = [annitem['retrievalfiles'][str(int(index)-1)]['filename'] for index in annitem['negativs']]

        dontknow_indices = [i for i in list(range(1,21)) if i not in annitem['annotations'] + annitem['negativs']]
        dknowfilenames = [annitem['retrievalfiles'][str(int(index)-1)]['filename'] for index in dontknow_indices]

        groundtruth_b_sg[b]['positives'][qname].extend(posfilenames)
        groundtruth_b_sg[b]['negatives'][qname].extend(negfilenames)
        groundtruth_b_sg[b]['dontknow'][qname].extend(dknowfilenames)
        groundtruth_sg['positives'][qname].extend(posfilenames)
        groundtruth_sg['negatives'][qname].extend(negfilenames)
        groundtruth_sg['dontknow'][qname].extend(dknowfilenames)

    stats += '\n' + "Scene Graphs Groundtruth Statistics:" + '\n'
    for qname in groundtruth_sg['positives'].keys():
        txt1 = "Number of relevant images for query {} = {}".format(qname, len(set(groundtruth_sg['positives'][qname])))
        txt2 = "Number of non-relevant images for query {} = {}".format(qname, len(set(groundtruth_sg['negatives'][qname])))
        print(txt1)
        print(txt2)
        stats += txt1 + '\n'
        stats += txt2 + '\n'

    print("-----------------------------------------------------")

    searchfolders_sg = {'scenegraph' : '/home/althausc/master_thesis_impl/retrieval/out/userstudy/scenegraphs/scenegraphs', 
                    'scenegraph-random' : '/home/althausc/master_thesis_impl/retrieval/out/userstudy/scenegraphs/scenegraphs_random'}

    results_precision_sg = {}

    for task, folder in searchfolders_sg.items():
        print('Calculation: {}'.format(task))
        results = defaultdict(list) #query image -> array of 0/1 (1 indicate a positive)
        results_graded = defaultdict(list) #query image -> array of 0/1/2 (0:negatives, 1:don't know, 2: positives)
        metadatadir = os.path.join(folder, 'metadata')
        onlyfiles = [os.path.join(metadatadir, f) for f in os.listdir(metadatadir) if os.path.isfile(os.path.join(metadatadir, f))]
        print(onlyfiles)
        print("Calculating metrics for {} files in directory {}.".format(len(onlyfiles), metadatadir))
        for fpath in onlyfiles:
            with open(fpath) as f:
                sranking = json.load(f)
            qname = os.path.splitext(os.path.basename(sranking['querypath']))[0]
            
            retrievalfiles = sorted(list(sranking['retrievaltopk'].items()), key= lambda x: int(x[0]))
            positives = [] # saves for each rank if image is positive (1) or not (0)
            graded = [] # saves for each rank 0,1,2, (0 for negatives and not seen images, 1= dontknow, 2= positives) 
            for r, item in retrievalfiles:
                rfilename = item['filename']
                if rfilename in groundtruth_sg['positives'][qname]:
                    positives.append(1)
                    graded.append(2)
                else:
                    positives.append(0)
                    if rfilename in groundtruth_sg['negatives'][qname]:
                        graded.append(0)
                    elif rfilename in groundtruth_sg['dontknow'][qname]:
                        graded.append(1)
                    else:
                        graded.append(0)
            results[qname].append(positives)
            results_graded[qname].append(graded)

        results_precision_sg[task] = getresults_queryimages(results, results_graded, groundtruth_sg, task)

    print("Results:")
    for task,metricresults in results_precision_sg.items():
        print(task)
        print(metricresults)

    filename = 'scenegraph-results.json'
    with open(os.path.join('/home/althausc/master_thesis_impl/results/userstudy', filename), 'w') as f:
        json.dump(results_precision_sg, f, indent=2)

    with open(os.path.join('/home/althausc/master_thesis_impl/results/userstudy', "stats.txt"), 'w') as f:
        f.write(stats)

def getresults_queryimages(qnames_results, qnames_gresults, groundtruth, task):
    p5s = []
    p10s = []
    p20s = []
    p50s = []
    aps = []
    for qname, positivesarray in qnames_results.items():
        for positives in positivesarray:
            n5 = np.sum(positives[:5])
            n10 = np.sum(positives[:10])
            n20 = np.sum(positives[:20])
            n50 = np.sum(positives[:50])
            p5s.append(n5/5)
            p10s.append(n10/10)
            p20s.append(n20/20)
            p50s.append(n50/50)

            print("----------")
            print("Positives: ", positives)
            print("----------")

            if len(groundtruth['positives'][qname])>0:
                pks = []
                for r,p in enumerate(positives, start=1):
                    if p == 1:
                        pks.append(np.sum(positives[:r])/r)
                ap = np.sum(pks)/len(groundtruth['positives'][qname]) #divided by all relevant documents per queryimage
                #print(positives)
                #print(np.sum(pks), len(groundtruth['positives'][qname]))
                aps.append(ap)

    print("Stats for descriptor/task ", task)
    print("P@5s", p5s)
    print("P@10s", p10s)
    print("P@20s", p20s)
    print("P@50s", p50s)
    print("Aps", aps)
    print("------------------------")
    # 0.1 = 0.1%
    p5 = "%.2f"%np.mean(p5s)
    p10 = "%.2f"%np.mean(p10s)
    p20 = "%.2f"%np.mean(p20s)
    p50 = "%.2f"%np.mean(p50s)
    mAp = "%.2f"%np.mean(aps)
    
    res = {"P@5": p5, "P@10": p10, "P@20": p20, "P@50": p50, 'mAP': mAp}

    ndcg5s = []
    ndcg10s = []
    ndcg20s = []
    ndcg50s = []

    for qname,gradedarray in qnames_gresults.items():
        for grades in gradedarray:
            dcg5 = np.sum([rel/math.log2(r+1) for r,rel in enumerate(grades[:5], start=1)])
            dcg10 = np.sum([rel/math.log2(r+1) for r,rel in enumerate(grades[:10], start=1)])
            dcg20 = np.sum([rel/math.log2(r+1) for r,rel in enumerate(grades[:20], start=1)])
            dcg50 = np.sum([rel/math.log2(r+1) for r,rel in enumerate(grades[:50], start=1)])
            


            idealordering = [2 for k in range(len(groundtruth['positives'][qname]))]
            idealordering += [1 for k in range(len(groundtruth['dontknow'][qname]))]
            idealordering += [0 for k in range(len(groundtruth['negatives'][qname]))]
            idealordering += [0 for k in range(50 - len(idealordering))]
            idcg5 = np.sum([rel/math.log2(r+1) for r,rel in enumerate(idealordering[:5], start=1)])
            idcg10 = np.sum([rel/math.log2(r+1) for r,rel in enumerate(idealordering[:10], start=1)])
            idcg20 = np.sum([rel/math.log2(r+1) for r,rel in enumerate(idealordering[:20], start=1)])
            idcg50 = np.sum([rel/math.log2(r+1) for r,rel in enumerate(idealordering[:50], start=1)])
            
            #if task == 'JLd_all_direct':

            print("----------------")
            print("Grades[:50]:", grades[:50])
            print("Ideal[:50]:", idealordering[:50])
            print(dcg50 , idcg50, dcg50/idcg50)
            print("----------------")

            ndcg5s.append(dcg5/idcg5)
            ndcg10s.append(dcg10/idcg10)
            ndcg20s.append(dcg20/idcg20)
            ndcg50s.append(dcg50/idcg50)

    print("Stats for descriptor/task ", task)
    print("nDCG5s:", ndcg5s)
    print("nDCG10s:", ndcg10s)
    print("nDCG20s:", ndcg20s)
    print("nDCG50s:", ndcg50s)
    print("------------------------")
    # 0.1 = 0.1%
    ndcg5 = "%.2f"%np.mean(ndcg5s)
    ndcg10 = "%.2f"%np.mean(ndcg10s)
    ndcg20 = "%.2f"%np.mean(ndcg20s)
    ndcg50 = "%.2f"%np.mean(ndcg50s)

    res.update({"nDCG@5": ndcg5, "nDCG@10": ndcg10, "nDCG@20": ndcg20, "nDCG@50": ndcg50})
    #exit(1)

    return res

if __name__ == "__main__":
    main()