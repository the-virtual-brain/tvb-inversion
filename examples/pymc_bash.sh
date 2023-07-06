
#!/bin/bash -l
#SBATCH --job-name="10node-pymc"
#SBATCH --account="ich012"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=name@domain
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --constraint=mc
#SBATCH --hint=multithread
#SBATCH --output=slurm.out

source ~/.bashrc
source activate tvb-env

stamp=$(date +%s)
export PYTENSOR_FLAGS="base_compiledir=/var/tmp/$stamp/.pytensor/,compile__timeout=24,compile__wait=10,device=cpu"

python3 ~/tvb-inversion/examples/pymc_inference_10nodes.py