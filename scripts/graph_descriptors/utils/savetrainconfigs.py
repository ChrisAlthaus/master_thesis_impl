import os
import shutil

#Save train configs of training the feature extractor


cpktfolders = [x[0] for x in os.walk('/home/althausc/master_thesis_impl/graph2vec/models')]
print(cpktfolders[:10])
outdir = '/home/althausc/master_thesis_impl/graph2vec/models/trainconfigs'


for runfolder in cpktfolders:
    if runfolder == '/home/althausc/master_thesis_impl/graph2vec/models/trainconfigs':
        continue

    if os.path.isfile(os.path.join(runfolder, 'config.txt')):
        newdir = os.path.join(outdir, os.path.basename(runfolder))
        os.makedirs(newdir)

        shutil.copy2(os.path.join(runfolder, 'config.txt'), newdir)
        if os.path.isfile(os.path.join(runfolder, '.logs', 'train_losses.csv')):
            shutil.copy2(os.path.join(runfolder, '.logs', 'train_losses.csv'), newdir)
        if os.path.isfile(os.path.join(runfolder, '.logs', 'val_losses.csv')):
            shutil.copy2(os.path.join(runfolder, '.logs', 'val_losses.csv'), newdir)
 
                 