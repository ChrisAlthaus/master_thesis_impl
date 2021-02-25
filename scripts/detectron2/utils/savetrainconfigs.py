import os
import shutil

#Save model training configuration and other important files, because some model folder deleted

cpktfolders = [x[0] for x in os.walk('/home/althausc/master_thesis_impl/detectron2/out/checkpoints')]
print(cpktfolders[:10])
outdir = '/home/althausc/master_thesis_impl/detectron2/out/checkpoints/trainconfigs'


for runfolder in cpktfolders:
    if runfolder == '/home/althausc/master_thesis_impl/detectron2/out/checkpoints/trainconfigs':
        continue

    if os.path.isfile(os.path.join(runfolder, 'configparams.txt')):
        newdir = os.path.join(outdir, os.path.basename(runfolder))
        os.makedirs(newdir)

        shutil.copy2(os.path.join(runfolder, 'configparams.txt'), newdir)
        if os.path.isfile(os.path.join(runfolder, 'model_architectur.txt')):
            shutil.copy2(os.path.join(runfolder, 'model_architectur.txt'), newdir)
        if os.path.isfile(os.path.join(runfolder, 'model_conf.txt')):
            shutil.copy2(os.path.join(runfolder, 'model_conf.txt'), newdir)
        
        onlyfiles = [f for f in os.listdir(runfolder) if os.path.isfile(os.path.join(runfolder, f))]
        for fname in onlyfiles:
            if 'events' in fname:
                 shutil.copy2(os.path.join(runfolder, fname), newdir)
                 break
 
                 