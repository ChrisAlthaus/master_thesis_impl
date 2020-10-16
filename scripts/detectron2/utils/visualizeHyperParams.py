import csv

csvfile = '/home/althausc/master_thesis_impl/detectron2/out/checkpoints/run_configs.csv'

_NETTYPE = 'ALL'

with open(csvfile, 'r', newline='') as csvFile:
    reader = csv.reader(csvFile, delimiter='\t')
    content = list(reader)


header = None  
dataout = []  

for i,row in enumerate(content):
    if i==0:
        header = [h.strip() for h in row]
    else:
        
        row = [r.strip() for r in row]
        print(row)
        lr = eval(row[header.index('LR')])
        batchsize = eval(row[header.index('ImPerBatch')])
        trainsize = eval(row[header.index('MinSize Train')])
        folder = row[header.index('Folder')]
        net = row[header.index('NET')]
        trainloss = eval(row[header.index('Train Loss')])
        valloss = eval(row[header.index('Val Loss')])
        bbox_ap = eval(row[header.index('bboxAP')])
        bbox_ap50 = eval(row[header.index('bboxAP50')])
        bbox_ap75 = eval(row[header.index('bboxAP75')])


        if net!= _NETTYPE:
            continue

        trainsize = trainsize[0] if len(trainsize)==1 else 1000 # 1000 for (640, 672, 704, 736, 768, 800)
        
        entry = {'x': trainsize, 'y': lr , 'z': batchsize, 'size': valloss, 'color': trainloss}
        text = 'MinSize Train: {}<br>LR: {}<br>Batchsize: {}<br>Folder: {}<br>Net: {}<br>TrainLoss: {}<br>ValLoss: {}<br>bboxAP: {}<br>bboxAP50: {}<br>bboxAP75: {}'\
                        .format(trainsize, lr, batchsize, folder, net, trainloss, valloss, bbox_ap, bbox_ap50, bbox_ap75)
        entry['text'] = text
        dataout.append(entry)

csvfile = '/home/althausc/master_thesis_impl/detectron2/out/checkpoints/plots/hyperparams_plotly.csv'
header = dataout[0].keys()
with open(csvfile, 'w', newline='') as csvFile:  
    writer = csv.writer(csvFile, delimiter='\t')
    writer.writerow(header)
    for entry in dataout:
        writer.writerow(entry.values())

print("Wrote hyper-parameter config for plotly to file: ", csvfile)

