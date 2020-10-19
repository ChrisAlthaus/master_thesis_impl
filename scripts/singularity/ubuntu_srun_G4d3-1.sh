#!/bin/bash
srun -G 4 -w devbox3 --pty singularity run --nv /home/althausc/images/ubuntu_200629.sif "$@"