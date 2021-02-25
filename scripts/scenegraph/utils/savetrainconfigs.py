import os
import shutil

#Save train configs of training the feature extractor


cpktfolders = [x[0] for x in os.walk('/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/faster_rcnn_training')]
print(cpktfolders[:10])
outdir = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/faster_rcnn_training/trainconfigs'


for runfolder in cpktfolders:
    if runfolder == '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch/checkpoints/faster_rcnn_training/trainconfigs':
        continue

    if os.path.isfile(os.path.join(runfolder, 'config.yml')):
        newdir = os.path.join(outdir, os.path.basename(runfolder))
        os.makedirs(newdir)

        shutil.copy2(os.path.join(runfolder, 'config.yml'), newdir)
        if os.path.isfile(os.path.join(runfolder, 'model_architectur.txt')):
            shutil.copy2(os.path.join(runfolder, 'model_architectur.txt'), newdir)
        if os.path.isfile(os.path.join(runfolder, 'log.txt')):
            shutil.copy2(os.path.join(runfolder, 'log.txt'), newdir)
        if os.path.isfile(os.path.join(runfolder, 'metrics.dat')):
            shutil.copy2(os.path.join(runfolder, 'metrics.dat'), newdir)
        if os.path.isfile(os.path.join(runfolder, 'paramsconfig.txt')):
            shutil.copy2(os.path.join(runfolder, 'paramsconfig.txt'), newdir)
        
        onlyfiles = [f for f in os.listdir(runfolder) if os.path.isfile(os.path.join(runfolder, f))]
        for fname in onlyfiles:
            if 'events' in fname:
                 shutil.copy2(os.path.join(runfolder, fname), newdir)
                 break
 
                 