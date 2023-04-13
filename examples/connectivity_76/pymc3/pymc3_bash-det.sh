#!/bin/bash -l
#SBATCH --job-name="76node-pymc3-det"
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
#SBATCH --hint=multithread
#SBATCH --mem=120GB
#SBATCH --output=pymc3_data/slurm-det.out

source ~/.bashrc
source activate tvb-env

export OMP_NUM_THREADS=16

stamp=$(date +%s)
export THEANO_FLAGS="base_compiledir=/var/tmp/$stamp/.theano/,compile__timeout=24,compile__wait=5,device=cpu,allow_gc=False"

python3 /users/erichter/tvb-inversion/tvb-inversion/examples/connectivity_76/pymc3/pymc3_inference_det.py