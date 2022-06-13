#!/bin/bash

#SBATCH --job-name=RepGAN_skip
#SBATCH --output=%x.o%j
#SBATCH --time=24:00:00
#SBATCH --error=error_skip.txt
#SBATCH --nodes=1
#SBATCH --mem=150gb
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1 
#SBATCH --partition=gpua100

export thisuser=$(whoami)
export hmd="/gpfs/users"
export wkd="/gpfs/workdir"
module purge
module load anaconda3/2021.05/gcc-9.2.0
module load cuda/11.4.0/gcc-9.2.0
source activate tf
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfs/users/colombergi/.conda/envs/tf/lib
 
python3 RepGAN_drive.py --nX 4000 --cuda --epochs 2000 --latentSdim 2 --latentNdim 20 --nXRepX 1 --nRepXRep 1 --nCritic 5 --nGenerator 1 --checkpoint_dir '/gpfs/workdir/colombergi/GiorgiaGAN/checkpoint/13_06' --results_dir '/gpfs/workdir/colombergi/GiorgiaGAN/results'
#python3 post_processing.py --nX 4000 --cuda --epochs 2000 --latentSdim 2 --latentNdim 20 --nXRepX 1 --nRepXRep 1 --nCritic 5 --nGenerator 1 --checkpoint_dir '/gpfs/workdir/colombergi/GiorgiaGAN/checkpoint/13_06' --results_dir '/gpfs/workdir/colombergi/GiorgiaGAN/results'