import os
import csv

cpktfolders = [x[0] for x in os.walk('/home/althausc/master_thesis_impl/graph2vec/models')]
print(cpktfolders[:10])


headers = ["mRank", "r20", "r50", "r100", "mScore", "NumDocs", "Epoch", "Vector dimensions", "Learning rate", "Min count", "Wl-iterations", "Epochs", "Steps Inference", "Min Feature Dimension", "Down-Sampling", "Descriptor Paths",'Number of graphs', "Folder"]
cfgdata = []



for runfolder in cpktfolders:
    print(runfolder)
    if os.path.isfile(os.path.join(runfolder, 'config.txt')) and os.path.isfile(os.path.join(runfolder, '.logs', 'val_losses.csv')) and not '0_false_ranks' in runfolder:
        dictdata =  {}
        with open(os.path.join(runfolder, 'config.txt'), "r") as f:
            for line in f:
                s = line.strip().split(":")
                if len(s)==2:
                    dictdata[s[0]] = s[1]
       
        cfgcurrent = []
        linetrain = None
        linevalidation = None
        with open(os.path.join(runfolder, '.logs', 'val_losses.csv'), 'r') as f:
            vlines = list(csv.reader(f, delimiter='\t'))
            if len(vlines)>1:
                linevalidation = vlines[-1]
            else:
                continue
        with open(os.path.join(runfolder, '.logs', 'train_losses.csv'), 'r') as f:
            tlines = list(csv.reader(f, delimiter='\t'))
            if len(tlines)>1:
                linetrain = tlines[-1]
            else:
                continue
        print(linetrain, linevalidation)
        linestogether = ['{}/{}'.format(x,y) for x,y in zip(linevalidation, linetrain)]
        cfgcurrent.extend([linestogether[2], linestogether[3], linestogether[4], linestogether[5], linestogether[1], linestogether[6], linestogether[0]])
        
        for h in headers[7:]:
            if h == "Folder":
                cfgcurrent.append(os.path.basename(runfolder))
                continue
            if h.strip() in dictdata:
                cfgcurrent.append(dictdata[h.strip()])
            else:
                cfgcurrent.append('x')
    
        cfgdata.append(cfgcurrent)

cfgdata.sort(key=lambda x: (int(x[headers.index("Number of graphs")]), int(x[headers.index("Vector dimensions")])))
with open(os.path.join('/home/althausc/master_thesis_impl/graph2vec/models', 'models-summary.csv'),'w') as csvfile:    
     writer = csv.writer(csvfile, delimiter=',')
     writer.writerow(headers)  

     for row in cfgdata:
        writer.writerow(row)