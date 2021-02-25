import os
import json
from collections import defaultdict
import numpy as np
import math
from nltk import agreement
import operator
from itertools import groupby

_DEBUG_PRECISION = False#True
_DEBUG_AGREEMENTS = False#True

def main():
    # --------------------------------------------------------- HUMAN POSES --------------------------------------------------
    print("HUMAN POSES")
    #Generate groundtruth annotation set
    filepaths1 = []
    annfolder = '/home/althausc/master_thesis_impl/flask/results/final/iart_results_new_2/iart_results_new_2'
    humanposefolders = ['JJo_reduced', 'Jc_rel', 'JcJLdLLa_reduced', 'JLd_all_direct', 'random'] 

    for descriptortype in humanposefolders:
        descriptordir = os.path.join(annfolder, descriptortype)
        onlyfiles = [os.path.join(descriptordir, f) for f in os.listdir(descriptordir) if os.path.isfile(os.path.join(descriptordir, f))]
        filepaths1.extend(onlyfiles)

    backgrounds = ['art_history', 'computer_science', 'other']
    groundtruth = {'positives': defaultdict(list), 'negatives': defaultdict(list), 'dontknow': defaultdict(list)}
    groundtruth_b = {}
    groundtruth_bh = {}
    qnametoindex_h = {}
    instancesindex = {}
    c_existingindices = 0
    
    bcounter = {'art_history':0, 'computer_science':0, 'other':0}
    users = []

    for b in backgrounds:
        groundtruth_bh[b] = {hp: {'positives': defaultdict(list), 'negatives': defaultdict(list), 'dontknow': defaultdict(list), 'users': defaultdict(list), 'retrievalfiles': defaultdict(list)} for hp in humanposefolders}
        qnametoindex_h.update({'JJo_reduced' : {}, 'Jc_rel': {}, 'JcJLdLLa_reduced' : {}, 'JLd_all_direct' : {}, 'random' : {}})

    if _DEBUG_PRECISION:
        print("Human Pose annotations:", filepaths1)
    print("Number of annotations", len(filepaths1))
    print()

    inst_index = 0
    filepaths1.sort()
    for fpath in filepaths1: #annotations
        with open(fpath) as f:
            annitem = json.load(f)
        qname = os.path.splitext(os.path.basename(annitem['query']))[0]
        b = annitem['background']

        posfilenames = [annitem['retrievalfiles'][str(int(index)-1)]['filename'] for index in annitem['annotations']]
        negfilenames = [annitem['retrievalfiles'][str(int(index)-1)]['filename'] for index in annitem['negativs']]

        dontknow_indices = [str(i) for i in list(range(1,21)) if str(i) not in annitem['annotations'] + annitem['negativs']]
        dknowfilenames = [annitem['retrievalfiles'][str(int(index)-1)]['filename'] for index in dontknow_indices]

        
        groundtruth['positives'][qname].extend(posfilenames)
        groundtruth['negatives'][qname].extend(negfilenames)
        groundtruth['dontknow'][qname].extend(dknowfilenames)
       
        for retrievalfile in list(annitem['retrievalfiles'].values())[:20]:
            instancename = "{}_{}".format(qname, retrievalfile['filename'])
            if instancename not in instancesindex:
                instancesindex[instancename] = len(instancesindex)
            else:
                c_existingindices += 1
        

        #For Agreements
        user = annitem['user']
        if user not in users:
            users.append(user)
            bcounter[b] += 1

        for h in humanposefolders:
            if h in fpath:
                if qname not in qnametoindex_h[h]:
                    #inst_index = (backgrounds.index(b) + 1) * humanposefolders.index(h)
                    qnametoindex_h[h].update({qname: inst_index})
                    inst_index += 1
                #for agreement calculations
                posindices = annitem['annotations']
                negindices = annitem['negativs']
                rfiles = [item['filename'] for item in list(annitem['retrievalfiles'].values())[:20]]
                if qname not in groundtruth_bh[b][h]['positives']:
                    groundtruth_bh[b][h]['positives'][qname] = [posindices] #to differentiate random and descriptors
                    groundtruth_bh[b][h]['negatives'][qname] = [negindices]
                    groundtruth_bh[b][h]['dontknow'][qname] = [dontknow_indices]
                    groundtruth_bh[b][h]['retrievalfiles'][qname] = [rfiles]
                else:
                    groundtruth_bh[b][h]['positives'][qname].append(posindices) #to differentiate random and descriptors
                    groundtruth_bh[b][h]['negatives'][qname].append(negindices)
                    groundtruth_bh[b][h]['dontknow'][qname].append(dontknow_indices)
                    groundtruth_bh[b][h]['retrievalfiles'][qname].append(rfiles)
                groundtruth_bh[b][h]['users'][qname].append(annitem['user'])
                break
        
               
    if _DEBUG_AGREEMENTS:
        print("qnametoindex_h:", qnametoindex_h)
        print("Index for instances (queryname + retrievalfilepath):", instancesindex)
        print("Double indices:", c_existingindices)
        print(groundtruth_bh['art_history']['JJo_reduced'])
   
    
    stats = "Human Pose Groundtruth Statistics:" + '\n'
    for qname in groundtruth['positives'].keys():
        txt1 = "Number of relevant images for query {} = {}".format(qname, len(set(groundtruth['positives'][qname])))
        txt2 = "Number of non-relevant images for query {} = {}".format(qname, len(set(groundtruth['negatives'][qname])))
        txt3 = "Number of dont know images for query {} = {}".format(qname, len(set(groundtruth['dontknow'][qname])))
        print(txt1)
        print(txt2)
        print(txt3)
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
    print("Users:", users)
    print("Background distribution:", bcounter)
    print("Results:")
    for descriptor,metricresults in results_precision.items():
        print(descriptor)
        for metric, results in metricresults.items():
            print(metric, results)

    results_agreement_hp = getresults_agreements(groundtruth_bh, instancesindex, users, backgrounds)

    print("Results:")
    print(results_agreement_hp)
    
    results_precision.update(results_agreement_hp)

    filename = 'humanpose-results.json'
    with open(os.path.join('/home/althausc/master_thesis_impl/results/userstudy', filename), 'w') as f:
        json.dump(results_precision, f, indent=2)

    #------------------------------------------------- SCENE GRAPHS --------------------------------------------------
    print("SCENE GRAPHS:")
    #Generate groundtruth annotation set
    filepaths2 = []
    annfolder = '/home/althausc/master_thesis_impl/flask/results/final/iart_results_new_2/iart_results_new_2'
    scenegraphfolders = ['scenegraphs', 'scenegraphs_random'] 

    for folder in scenegraphfolders:
        descriptordir = os.path.join(annfolder, folder)
        onlyfiles = [os.path.join(descriptordir, f) for f in os.listdir(descriptordir) if os.path.isfile(os.path.join(descriptordir, f))]
        filepaths2.extend(onlyfiles)

    if _DEBUG_AGREEMENTS:
        print("Scene graph annotations", filepaths2)
    print("Number of annotations", len(filepaths2))
    print()

    backgrounds_sg = ['art_history', 'computer_science', 'other']
    groundtruth_sg = {'positives': defaultdict(list), 'negatives': defaultdict(list), 'dontknow': defaultdict(list)}
    groundtruth_b_sg = {}
    for b in backgrounds_sg:
        groundtruth_b_sg.update({b: {'positives': defaultdict(list), 'negatives': defaultdict(list), 'dontknow': defaultdict(list)}})

        groundtruth_bh[b].update({sf: {'positives': defaultdict(list), 'negatives': defaultdict(list), 'dontknow': defaultdict(list), 'users': defaultdict(list)} for sf in scenegraphfolders})
        qnametoindex_h.update({'scenegraphs' : {}, 'scenegraphs_random': {}})
        

    if _DEBUG_AGREEMENTS:
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

        user = annitem['user']
        if user not in users:
            users.append(user)

        for s in scenegraphfolders:
            if s in fpath:
                if qname not in qnametoindex_h[s]:
                    #inst_index = (backgrounds.index(b) + 1) * humanposefolders.index(h)
                    #print(qname)
                    qnametoindex_h[s].update({qname: inst_index})
                    inst_index += 1
                #for agreement calculations
                posindices = annitem['annotations']
                negindices = annitem['negativs']
                if qname not in groundtruth_bh[b][s]['positives']:
                    groundtruth_bh[b][s]['positives'][qname] = [posindices] #to differentiate random and descriptors
                    groundtruth_bh[b][s]['negatives'][qname] = [negindices]
                    groundtruth_bh[b][s]['dontknow'][qname] = [dontknow_indices]
                else:
                    groundtruth_bh[b][s]['positives'][qname].append(posindices) #to differentiate random and descriptors
                    groundtruth_bh[b][s]['negatives'][qname].append(negindices)
                    groundtruth_bh[b][s]['dontknow'][qname].append(dontknow_indices)
                groundtruth_bh[b][s]['users'][qname].append(annitem['user'])
                break

    stats += '\n' + "Scene Graphs Groundtruth Statistics:" + '\n'
    for qname in groundtruth_sg['positives'].keys():
        txt1 = "Number of relevant images for query {} = {}".format(qname, len(set(groundtruth_sg['positives'][qname])))
        txt2 = "Number of non-relevant images for query {} = {}".format(qname, len(set(groundtruth_sg['negatives'][qname])))
        txt3 = "Number of dont know images for query {} = {}".format(qname, len(set(groundtruth_sg['dontknow'][qname])))
        print(txt1)
        print(txt2)
        print(txt3)
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

# --------------------------------------------------------------------------------------------------------------------
    #Save agreements of scene graph and human poses
    #results_agreement_hpsg = getresults_agreements(groundtruth_bh, qnametoindex_h, users, backgrounds)
   # print("Results:")
    #print(results_agreement_hpsg)
        
    #filename = 'agreements-hpandsg-results.json'
    #with open(os.path.join('/home/althausc/master_thesis_impl/results/userstudy', filename), 'w') as f:
    #    json.dump(results_agreement_hpsg, f, indent=2)

    #Save statistics
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


            if _DEBUG_PRECISION:
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

    if _DEBUG_PRECISION:
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
            if _DEBUG_PRECISION:
                print("----------------")
                print("Grades[:50]:", grades[:50])
                print("Ideal[:50]:", idealordering[:50])
                print(dcg50 , idcg50, dcg50/idcg50)
                print("----------------")

            ndcg5s.append(dcg5/idcg5)
            ndcg10s.append(dcg10/idcg10)
            ndcg20s.append(dcg20/idcg20)
            ndcg50s.append(dcg50/idcg50)

    if _DEBUG_PRECISION:
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

def getresults_agreements(groundtruth_bh, instancesindex, users, backgrounds):
    codings_artcomputer = []
    codings_artothers = []
    codings_computerothers = []
    codingsb = []
    codingsusers = []
    #['art_history', 'computer_science', 'others']
    for b,hs in groundtruth_bh.items():
        for h, annotations in hs.items():
            for (qname1, annlists), (qname2, rfileslists) in zip(annotations['positives'].items(), annotations['retrievalfiles'].items()):
                assert qname1 == qname2
                #[coder,instance,code]
                coderb = backgrounds.index(b)
                #instance = qnametoindex_h[h][qname] #background->descriptor->qname->instance_num_of_ranking
                #instance = instancesindex[qname]
                users_qname = annotations['users'][qname1]
                
                for i, (annlist, rfiles) in enumerate(zip(annlists, rfileslists)):
                    for c in annlist:
                        assert int(c)>0 and int(c)<=20
                        instancename = "{}_{}".format(qname1, rfiles[int(c)-1])
                        instance = instancesindex[instancename]

                        codingsb.append([coderb, instance, 2])
                        if b == 'art_history': 
                            codings_artcomputer.append([coderb, instance , 2])
                            codings_artothers.append([coderb, instance, 2])
                        elif b == 'computer_science':
                            codings_artcomputer.append([coderb, instance, 2])
                            codings_computerothers.append([coderb, instance, 2])
                        elif b == 'other':
                            codings_artothers.append([coderb, instance , 2])
                            codings_computerothers.append([coderb, instance, 2])
                        else:
                            raise ValueError()
                        codingsusers.append([users.index(users_qname[i]), instance, 2])
                
            for (qname1, annlists), (qname2, rfileslists) in zip(annotations['negatives'].items(), annotations['retrievalfiles'].items()):
                assert qname1 == qname2
                #[coder,instance,code]
                coderb = backgrounds.index(b)
                #instance = qnametoindex_h[h][qname] #background->descriptor->qname->instance_num_of_ranking
                #instance = instancesindex[qname]
                users_qname = annotations['users'][qname1]
                
                for i, (annlist, rfiles) in enumerate(zip(annlists, rfileslists)):
                    for c in annlist:
                        assert int(c)>0 and int(c)<=20
                        instancename = "{}_{}".format(qname1, rfiles[int(c)-1])
                        instance = instancesindex[instancename]

                        codingsb.append([coderb, instance, 0])
                        if b == 'art_history': 
                            codings_artcomputer.append([coderb, instance , 0])
                            codings_artothers.append([coderb, instance, 0])
                        elif b == 'computer_science':
                            codings_artcomputer.append([coderb, instance, 0])
                            codings_computerothers.append([coderb, instance, 0])
                        elif b == 'other':
                            codings_artothers.append([coderb, instance , 0])
                            codings_computerothers.append([coderb, instance, 0])
                        else:
                            raise ValueError()
                        codingsusers.append([users.index(users_qname[i]), instance, 0])

            for (qname1, annlists), (qname2, rfileslists) in zip(annotations['dontknow'].items(), annotations['retrievalfiles'].items()):
                assert qname1 == qname2
                #[coder,instance,code]
                coderb = backgrounds.index(b)
                #instance = qnametoindex_h[h][qname] #background->descriptor->qname->instance_num_of_ranking
                #instance = instancesindex[qname]
                users_qname = annotations['users'][qname1]
                
                for i, (annlist, rfiles) in enumerate(zip(annlists, rfileslists)):
                    for c in annlist:
                        assert int(c)>0 and int(c)<=20
                        instancename = "{}_{}".format(qname1, rfiles[int(c)-1])
                        instance = instancesindex[instancename]

                        codingsb.append([coderb, instance, 1])
                        if b == 'art_history': 
                            codings_artcomputer.append([coderb, instance , 1])
                            codings_artothers.append([coderb, instance, 1])
                        elif b == 'computer_science':
                            codings_artcomputer.append([coderb, instance, 1])
                            codings_computerothers.append([coderb, instance, 1])
                        elif b == 'other':
                            codings_artothers.append([coderb, instance , 1])
                            codings_computerothers.append([coderb, instance, 1])
                        else:
                            raise ValueError()
                        codingsusers.append([users.index(users_qname[i]), instance, 1])
    
    #stats
    counter = defaultdict(int)
    for item in codingsusers:
        counter[item[0]] += 1

    
    print("Users for per-user agreement:", users)
    print("Number annotations/ user:" ,counter)
  
    usersperinstance = defaultdict(list)
    for item in codingsusers:
        usersperinstance[item[1]].append(item[0])
    c = 0
    c50 = 0
    for instance, vusers in usersperinstance.items():
        if len(users) == len(vusers):
            c += 1
        if len(vusers) <= len(users)*0.5: 
            c50 += 1

    print("Number of overlapping instances which were annotated by all users:", c)
    print("Number of overlapping instances which were annotated by at least >=0.5 users:", c50)
    print("Number of total instances:", len(usersperinstance))

    #stats end
    codingcollection = {'betweenusers': codingsusers, 'betweenbackgrounds': codingsb, 'artcomputer':codings_artcomputer, 'artothers':codings_artothers, 'computerothers':codings_computerothers}
    results = {}
    for ctype, codingset in codingcollection.items():
        ratingtask = agreement.AnnotationTask(data=codingset)
        try:
            results[ctype] = {'Krippendorff\'s alpha:':ratingtask.alpha(), 'Scott\'s pi:' :ratingtask.pi(), 'Cohen\'s Kappa:':ratingtask.kappa()}
        except ZeroDivisionError as e:
            print(e)
            print("Warning may because no annotation of one background type. Background current:", ctype)

    return results

if __name__ == "__main__":
    main()