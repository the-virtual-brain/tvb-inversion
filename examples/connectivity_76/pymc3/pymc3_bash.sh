#!/bin/bash -l
#SBATCH --job-name="2node-limitcycle-pymc"
#SBATCH --account="ich012"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=emilius.richter@fu-berlin.de
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --constraint=mc
#SBATCH --hint=nomultithread
#SBATCH --output=pymc_data/slurm.out

source ~/.bashrc
source activate tvb-env

export OMP_NUM_THREADS=16

theano-cache purge
stamp=$(date +%s)
export THEANO_FLAGS="base_compiledir=/var/tmp/$stamp/.theano/,compile__timeout=24,compile__wait=10,device=cpu"

python3 /users/erichter/tvb-inversion/tvb-inversion/examples/connectivity_76/pymc3/pymc3_inference.py