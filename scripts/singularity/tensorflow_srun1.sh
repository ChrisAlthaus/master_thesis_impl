#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:t2080ti:1
srun singularity run --nv  /nfs/data/env/tensorflow_1.15_gpu_althaus.sif "$@"