#!/bin/bash -l
#SBATCH --job-name="76node-sbi"
#SBATCH --account="ich012"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=emilius.richter@fu-berlin.de
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --constraint=mc
#SBATCH --hint=multithread
#SBATCH --output=sbi_data/slurm.out

source ~/.bashrc
source activate tvb-env

python3 /users/erichter/tvb-inversion/tvb-inversion/examples/connectivity_76/sbi/sbi_inference.py
