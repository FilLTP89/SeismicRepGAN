#!/bin/bash

#SBATCH --job-name=load
#SBATCH --output=%x.o%j
#SBATCH --time=168:00:00
#SBATCH --error=error_load.txt
#SBATCH --mem=150gb
#SBATCH --partition=cpu_long

 
python3 MDOFload_prova.py