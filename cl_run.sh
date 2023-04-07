#!/bin/bash

# python3 RepGAN.py --dataroot /home/kltpk89/Data/Filippo/aeolus/mdof/damaged_1_8P --nX 512 --cuda --epochs 400 --kernel 3 --stride 2 --nAElayers 3 --nZfirst 8 --latentSdim 256 --latentNdim 64 --nSlayers 1 --nClayers 1 --nNlayers 1 --Sstride 2 --Cstride 2 --Nstride 2 --Sstride 2 --Skernel 3 --Ckernel 3 --Nkernel 3

python3 RepGAN_drive.py --nX 200 --cuda --epochs 2000 --kernel 3 --stride 2 \
    --nAElayers 3 --nZfirst 8 --latentSdim 2 --latentNdim 2 --nSlayers 1  \
    --nClayers 1 --nNlayers 1 --Sstride 2 --Cstride 2 --Nstride 2 --Sstride 2 \
    --Skernel 3 --Ckernel 3 --Nkernel 3 \
    --dataroot_1 /home/kltpk89/Data/Filippo/aeolus/stead_1_1U \
    --dataroot_2 /home/kltpk89/Data/Filippo/aeolus/stead_1_1D
