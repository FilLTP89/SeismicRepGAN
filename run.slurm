#!/bin/bash
#SBATCH --job-name=RepGAN
#SBATCH --output=%x.o%j
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Module load
export thisuser=$(whoami)
export hmd="/gpfs/users"
module purge
module load anaconda3/2020.02/gcc-9.2.0
module load cuda/10.1.243/intel-19.0.3.199
module load arm-forge/20.1.3-Redhat-7.0/intel-19.0.3.199

source activate tf
export LD_LIBRARY_PATH=/gpfs/workdir/invsem07/.conda/envs/tf/lib:$LD_LIBRARY_PATH
# export ALLINEA_MPI_INIT=main 
# export ALLINEA_HOLD_MPI_INIT=1
#export ALLINEA_FORCE_CUDA_VERSION=10.1
# export ALLINEA_STOP_AT_MAIN=1

application=`which python3`
#/gpfs/softs/languages/anaconda3/2020.02/bin/python3
ddt --connect --mem-debug=balanced srun -n 1  $application %allinea_python_debug% RepGAN.py
