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

python3 RepGAN_drive.py --nX 4000 --cuda --epochs 2000 --latentSdim 2 --latentNdim 256 --nXRepX 1 --nRepXRep 2 --nCritic 1 --nGenerator 5 --nSlayers 1 --nNlayers 1 --nClayers 1 --DxTrainType WGAN --DsTrainType WGAN --DcTrainType WGAN --DnTrainType WGAN --DxSN --DzSN --FxSN --GzSN --DxLR 0.000002 --DsLR 0.00001 --DnLR 0.00001 --DcLR 0.00001 --FxLR 0.000002 --GzLR 0.000002 --checkpoint_dir '/gpfs/workdir/colombergi/GiorgiaGAN/checkpoint/14_07' --results_dir '/gpfs/workdir/colombergi/GiorgiaGAN/results'
#python3 post_processing.py --nX 4000 --cuda --epochs 2 --latentSdim 2 --latentNdim 256 --nXRepX 1 --nRepXRep 2 --nCritic 1 --nGenerator 5 --nSlayers 1 --nNlayers 1 --nClayers 1 --DxLR 0.000002 --DsLR 0.00001 --DnLR 0.00001 --DcLR 0.00001 --FxLR 0.000002 --GzLR 0.000002 --checkpoint_dir '/gpfs/workdir/colombergi/GiorgiaGAN/checkpoint/14_07' --results_dir '/gpfs/workdir/colombergi/GiorgiaGAN/results'