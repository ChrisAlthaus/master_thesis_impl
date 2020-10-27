import csv

#Script is used to summarize all the runs in rum_configs.csv by creating a text csv.
#This csv is used to visualize the hyperparameters in the application plotly.

csvfile = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/run_configs.csv'
_PREDICTORS = ['MotifPredictor', 'IMPPredictor', 'VCTreePredictor', 'TransformerPredictor', 'CausalAnalysisPredictor']
predictor = _PREDICTORS[0]

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
        batchsize = eval(row[header.index('Batchsize')])
        backbone = row[header.index('FasterRCNN')]
        fusiontype = row[header.index('Fusion')]
        contextlayer = row[header.index('ContextLayer')]
        minsize = eval(row[header.index('MinSize')])
        dataset = row[header.index('Dataset')]
        
        try:
            trainloss = eval(row[header.index('Train Loss')])
            loss_refined = eval(row[header.index('Loss_refined')])
            loss_rel = eval(row[header.index('Loss_rel')])
            recall = eval(row[header.index('R@100')]) if row[header.index('R@100')] != 'not found' else 0.0
        except (ValueError, SyntaxError) as e:
            print("Entry {} cannot be processed, because important loss values are missing.".format(row))
            continue
        folder = row[header.index('Folder')]


        if row[header.index('Predictor')] == _PREDICTORS[0]:
            entry = {'x': lr , 'y': contextlayer , 'z': fusiontype, 'size': recall, 'color': trainloss}
            text = 'MinSize Train: {}<br>LR: {}<br>Batchsize: {}<br>Folder: {}<br>Backbone: {}<br>Dataset: {}<br>TrainLoss: {}<br>TrainLoss_refined: {}<br>TrainLoss_rel: {}<br>R@100: {}'\
                            .format(minsize, lr, batchsize, folder, backbone, dataset, trainloss, loss_refined, loss_rel, recall)
            entry['text'] = text
            dataout.append(entry)
        else:
            entry = {'x': lr , 'y': batchsize , 'z': predictor, 'size': recall, 'color': trainloss}
            text = 'MinSize Train: {}<br>LR: {}<br>Batchsize: {}<br>Folder: {}<br>Backbone: {}<br>Dataset: {}<br>TrainLoss: {}<br>TrainLoss_refined: {}<br>TrainLoss_rel: {}<br>R@100: {}'\
                            .format(minsize, lr, batchsize, folder, backbone, dataset, trainloss, loss_refined, loss_rel, recall)
            entry['text'] = text
            dataout.append(entry) 
        print("--------------------------")


csvfile = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/plot/hyperparams_plotly_{}.csv'.format(predictor)
header = dataout[0].keys()
with open(csvfile, 'w', newline='') as csvFile:  
    writer = csv.writer(csvFile, delimiter='\t')
    writer.writerow(header)
    for entry in dataout:
        writer.writerow(entry.values())

print("Wrote hyper-parameter config for plotly to file: ", csvfile)

