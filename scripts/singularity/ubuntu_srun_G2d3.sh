#!/bin/bash
srun -G 2 -w devbox3 --pty singularity run --nv /home/althausc/images/ubuntu_200629.sif "$@"