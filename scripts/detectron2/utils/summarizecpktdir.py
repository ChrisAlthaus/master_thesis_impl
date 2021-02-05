import os
import csv

cpktfolders = [x[0] for x in os.walk('/home/althausc/master_thesis_impl/detectron2/out/checkpoints')]
print(cpktfolders[:10])


headers = ["bbox/AP50", "keypoints/AP50", "trainmode", "dataaugm", "batchsize", "epochs", "lr", "bn", "minscales", "minkpt", "time", "addnotes", "folder"]
cfgdata = []

for runfolder in cpktfolders:
    if os.path.isfile(os.path.join(runfolder, 'configparams.txt')) and os.path.isfile(os.path.join(runfolder, 'metrics.json')):
        txtstr = open(os.path.join(runfolder, 'configparams.txt'), 'r').read().replace('true', str(True)).replace('false', str(False))
        config = eval(txtstr)

        metrics = open(os.path.join(runfolder, 'metrics.json')).readlines()
        if len(metrics)<=100:
            continue
        bboxscore = None
        kptscore = None
        firstline = eval(metrics[0])
        time = firstline["eta_seconds"]/60/60 if "eta_seconds" in firstline else 'not known'
        for line in reversed(metrics):
            if "bbox/AP50" in eval(line.replace("NaN",'0.0')):
                bboxscore = eval(line.replace("NaN",'0.0'))["bbox/AP50"]
                kptscore = eval(line.replace("NaN",'0.0'))["keypoints/AP50"] 
                if bboxscore != 'NaN':
                    break

        folder = os.path.basename(runfolder)
        if not bboxscore:
            bboxscore = 0.0
        if not kptscore:
            kptscore = 0.0
        cfgdata.append([bboxscore, kptscore, config["trainmode"], config["dataaugm"], config["batchsize"], config["epochs"], config["lr"], config["bn"], config["minscales"], config["minkpt"], time, config["addnotes"], folder])

cfgdata.sort(key=lambda x: (x[headers.index("minscales")], x[headers.index("bbox/AP50")], x[headers.index("keypoints/AP50")]))
with open(os.path.join('/home/althausc/master_thesis_impl/detectron2/out', 'cptks-summary.csv'),'w') as csvfile:    
     writer = csv.writer(csvfile, delimiter=',')
     writer.writerow(headers)  

     for row in cfgdata:
        writer.writerow(row)