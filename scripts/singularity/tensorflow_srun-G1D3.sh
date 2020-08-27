#!/bin/bash
srun -G 1 -w devbox3 --pty singularity run --nv /nfs/data/env/tensorflow_1.15_gpu_althaus.sif "$@"