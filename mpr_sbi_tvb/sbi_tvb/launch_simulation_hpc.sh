#!/bin/sh

#echo "Pulling tvb-run:inv docker image"
/apps/daint/system/opt/sarus/1.5.1/bin/sarus pull thevirtualbrain/tvb-run:inv-sbi

start=$SECONDS

echo "Start sarus container"
/apps/daint/system/opt/sarus/1.5.1/bin/sarus --debug run --mount=type=bind,source=$PWD,destination=$1 thevirtualbrain/tvb-run:inv-sbi /bin/bash -c "/opt/conda/envs/tvb-run/bin/python -m simulation_hpc_launcher $2 $3 $4"

duration=$(( SECONDS - start ))

echo "TVB operation run completed in $duration seconds"
