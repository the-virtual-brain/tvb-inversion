#!/bin/sh

#echo "Pulling tvb-run:inv docker image"
#docker pull thevirtualbrain/tvb-run:inv

start=$SECONDS

echo "Start docker container"
docker run -v $1:/home/data thevirtualbrain/tvb-run:inv /bin/bash -c "/opt/conda/envs/tvb-run/bin/python -m tvb.core.simulation_hpc_launcher $2"

duration=$(( SECONDS - start ))

echo "TVB operation run completed in $duration seconds"
