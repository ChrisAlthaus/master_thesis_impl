# Run this script in the detectron2 base folder

#python3 -m pip install -e detectron2
srun -G 1 -w srun -G 1 -w devbox3 --pty singularity run --nv  ../../images/ubuntu_200629.sif \
python3.6 -m pip install --user detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/index.html

mkdir out
