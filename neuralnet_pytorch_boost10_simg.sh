#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU
#SBATCH --ntasks-per-node 28
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:p100:2

module load AI/anaconda3-5.1.0_gpu.2018-08
source activate $AI_ENV

SIMG=/pylon5/containers/ngc/pytorch/19.10-py3.simg
S_EXEC="singularity exec --nv ${SIMG}"

date

python /pylon5/as5fphp/sidd529/projects/lya/neuralnet_pytorch_boost10.py