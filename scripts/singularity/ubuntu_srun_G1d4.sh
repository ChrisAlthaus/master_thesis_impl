#!/bin/bash
srun -G 1 -w devbox4 --pty singularity run --nv /home/althausc/images/ubuntu_200914.sif "$@"