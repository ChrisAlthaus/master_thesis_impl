#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:t2080ti:1
srun singularity run --nv  /home/althausc/images/ubuntu_200914.sif "$@"
