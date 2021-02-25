import subprocess
import os

#Save repositories directory structure for seeing added directories later
outdir = '/home/althausc/master_thesis_impl/scripts/repofolder_layouts'
#Scene Graph Benchmark 
repodir = '/home/althausc/master_thesis_impl/Scene-Graph-Benchmark.pytorch'
command = 'bash tree.sh {}'.format(repodir)
process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

#outdir = '/home/althausc/master_thesis_impl/scripts/scenegraph'
with open(os.path.join(outdir, ".repository-scenegraph"),"w") as f:
    f.write(output.decode('ascii'))
    print("Created File: ",os.path.join(outdir, ".repository-scenegraph"))

#Mask-RCNN
repodir = '/home/althausc/master_thesis_impl/detectron2'
command = 'bash tree.sh {}'.format(repodir)
process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

#outdir = '/home/althausc/master_thesis_impl/scripts/detectron2'
with open(os.path.join(outdir, ".repository-detectron2"),"w") as f:
    f.write(output.decode('ascii'))
    print("Created File: ",os.path.join(outdir, ".repository-detectron2"))


#PoseFix
repodir = '/home/althausc/master_thesis_impl/PoseFix_RELEASE'
command = 'bash tree.sh {}'.format(repodir)
process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

#outdir = '/home/althausc/master_thesis_impl/scripts/posefix'
with open(os.path.join(outdir, ".repository-posefix"),"w") as f:
    f.write(output.decode('ascii'))
    print("Created File: ",os.path.join(outdir, ".repository-posefix"))



#Workflow Directories

#Pose Descriptors Storage?!
repodir = '/home/althausc/master_thesis_impl/posedescriptors'
command = 'bash tree.sh {}'.format(repodir)
process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

#outdir = '/home/althausc/master_thesis_impl/scripts/pose_descriptors'
with open(os.path.join(outdir, ".directory-posedescriptors"),"w") as f:
    f.write(output.decode('ascii'))
    print("Created File: ",os.path.join(outdir, ".directory-posedescriptors"))


#Evaluation files & Jupyter notebook logs
repodir = '/home/althausc/master_thesis_impl/results'
command = 'bash tree.sh {}'.format(repodir)
process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

#outdir = '/home/althausc/master_thesis_impl/scripts/branchtogether'
with open(os.path.join(outdir, ".directory-results"),"w") as f:
    f.write(output.decode('ascii'))
    print("Created File: ",os.path.join(outdir, ".directory-results"))


#Retrieval top-k Files & Elastic Search Config
repodir = '/home/althausc/master_thesis_impl/retrieval'
command = 'bash tree.sh {}'.format(repodir)
process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

#outdir = '/home/althausc/master_thesis_impl/scripts/branchtogether'
with open(os.path.join(outdir, ".directory-retrieval"),"w") as f:
    f.write(output.decode('ascii'))
    print("Created File: ",os.path.join(outdir, ".directory-retrieval"))


#Graph2Vec
repodir = '/home/althausc/master_thesis_impl/graph2vec'
command = 'bash tree.sh {}'.format(repodir)
process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

#outdir = '/home/althausc/master_thesis_impl/scripts/branchtogether'
with open(os.path.join(outdir, ".directory-graph2vec"),"w") as f:
    f.write(output.decode('ascii'))
    print("Created File: ",os.path.join(outdir, ".directory-graph2vec"))


#User study
repodir = '/home/althausc/master_thesis_impl/flask'
command = 'bash tree.sh {}'.format(repodir)
process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

#outdir = '/home/althausc/master_thesis_impl/scripts/branchtogether'
with open(os.path.join(outdir, ".directory-userstudy"),"w") as f:
    f.write(output.decode('ascii'))
    print("Created File: ",os.path.join(outdir, ".directory-userstudy"))