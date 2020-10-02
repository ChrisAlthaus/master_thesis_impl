#!/bin/bash
srun -G 1 -w devbox4 --pty singularity run --nv /home/althausc/images/ubuntu_200629.sif "$@"    #for Mask-RCNN