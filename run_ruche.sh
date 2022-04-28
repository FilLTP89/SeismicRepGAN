#!/bin/bash

#SBATCH --job-name=RepGAN_skip
#SBATCH --output=%x.o%j
#SBATCH --time=24:00:00
#SBATCH --error=error_skip.txt
#SBATCH --nodes=1
#SBATCH --mem=512gb
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1 
#SBATCH --partition=gpu

export thisuser=$(whoami)
export hmd="/gpfs/users"
export wkd="/gpfs/workdir"
module purge
module load anaconda3/2020.02/gcc-9.2.0 cuda/10.1.243/intel-19.0.3.199
source activate tf
export LD_LIBRARY_PATH=$wkd/$thisuser/.conda/envs/tf/lib:$LD_LIBRARY_PATH
 
python3 RepGAN_skip.py --nX 4000 --cuda --epochs 2 --kernel 3 --stride 2 --nAElayers 3 --nZfirst 8 --latentSdim 2 --latentNdim 20 --nSlayers 1 --nClayers 1 --nNlayers 1 --Sstride 2 --Cstride 2 --Nstride 2 --Sstride 2 --Skernel 3 --Ckernel 3 --Nkernel 3
#python3 post_processing_skip.py