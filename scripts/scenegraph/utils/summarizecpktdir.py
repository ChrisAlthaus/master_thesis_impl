import os
import csv

#Summarize training configurations and evaluation scores in .csv table

_MODE = 'SGDETTRAIN'
if _MODE == 'BACKBONE':
    cpktfolders = [x[0] for x in os.walk('/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/faster_rcnn_training/logs')]
    print(cpktfolders[:10])

    headers = ["mAP", "datasetselect",  "trainbatchsize", "gpus", "lr", "maxiterations", "addnotes", "folder"]
    cfgdata = []

    for runfolder in cpktfolders:
        if os.path.isfile(os.path.join(runfolder, 'paramsconfig.txt')) and os.path.isfile(os.path.join(runfolder, 'train.log')):
            txtstr = open(os.path.join(runfolder, 'paramsconfig.txt'), 'r').read().replace('true', str(True)).replace('false', str(False))
            config = eval(txtstr)

            rowdata = []
            for h in headers[:-1]:
                if h in config:
                    rowdata.append(config[h])
                else:
                    rowdata.append("x")
            
            folder = os.path.basename(runfolder)
            rowdata.append(folder)

            cfgdata.append(rowdata)

    cfgdata.sort(key=lambda x: (x[headers.index("folder")]))
    print("Write to folder: ", '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/faster_rcnn_training/logs')
    with open(os.path.join('/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/faster_rcnn_training/logs', 'cptks-summary.csv'),'w') as csvfile:    
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(headers)  

        for row in cfgdata:
            writer.writerow(row)
else:
    cpktfolders = [x[0] for x in os.walk('/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/sgdet_training/logs')]
    print(cpktfolders[:10])
    headers = ["mAP", "datasetselect",  "trainbatchsize", "gpus", "lr", "predictor", "fusiontype", "contextlayer", "maxiterations", "pretrainedcpkt", "addnotes", "folder"]
    cfgdata = []

    for runfolder in cpktfolders:
        if os.path.isfile(os.path.join(runfolder, 'paramsconfig.txt')) and os.path.isfile(os.path.join(runfolder, 'train.log')):
            txtstr = open(os.path.join(runfolder, 'paramsconfig.txt'), 'r').read().replace('true', str(True)).replace('false', str(False))
            config = eval(txtstr)

            rowdata = []
            for h in headers[:-1]:
                if h in config:
                    rowdata.append(config[h])
                else:
                    rowdata.append("x")
            
            folder = os.path.basename(runfolder)
            rowdata.append(folder)

            cfgdata.append(rowdata)

    cfgdata.sort(key=lambda x: (x[headers.index("folder")]))
    print("Write to folder: ", '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/sgdet_training/logs')
    with open(os.path.join('/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/sgdet_training/logs', 'cptks-summary.csv'),'w') as csvfile:    
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(headers)  

        for row in cfgdata:
            writer.writerow(row)