#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:t2080ti:2
srun singularity run --nv  /home/althausc/images/ubuntu_200629.sif "$@"
