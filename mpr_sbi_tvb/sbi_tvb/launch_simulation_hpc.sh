#!/bin/sh

#echo "Pulling tvb-run:inv docker image"
module load sarus && sarus pull thevirtualbrain/tvb-run:inv-sbi

start=$SECONDS

echo "Start docker container"
srun -C mc sarus --debug run --mount=type=bind,source=$PWD,destination=$1 thevirtualbrain/tvb-run:inv-sbi /bin/bash -c "/opt/conda/envs/tvb-run/bin/python -m tvb.core.simulation_hpc_launcher $2"

duration=$(( SECONDS - start ))

echo "TVB operation run completed in $duration seconds"
