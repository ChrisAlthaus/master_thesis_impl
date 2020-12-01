#!/bin/bash
srun -G 1 -w devbox3 --pty singularity run --nv /home/althausc/images/ubuntu_200914.sif "$@"   #for Scene Graphs