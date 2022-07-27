#!/bin/sh

#echo "Pulling tvb-run:inv docker image"
docker pull thevirtualbrain/tvb-run:inv-sbi

start=$SECONDS

echo "Start docker container"
docker run -v $1:$2 thevirtualbrain/tvb-run:inv-sbi /bin/bash -c "/opt/conda/envs/tvb-run/bin/python -m simulation_hpc_launcher $3 $4 $5"

duration=$(( SECONDS - start ))

echo "TVB operation run completed in $duration seconds"
