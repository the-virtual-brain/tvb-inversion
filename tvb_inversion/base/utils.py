from typing import List
import os

from tvb_inversion.base.proc_exec import \
    JobLibExec, PostProcess, SaveMetricsToDisk, Reduction, DaskExec
from tvb_inversion.base.metrics import Metric
from tvb_inversion.base.sim_seq import SimSeq


def data_path(relp):
    data_root = os.path.abspath(
            os.path.join(
                os.path.dirname(
                    os.path.dirname(__file__)
                ),
                'data_input_files'
            )
    )
    return os.path.join(data_root, os.path.normpath(relp))


def run_local(seq: SimSeq, metrics: List[Metric], filename='results',
              reduction: Reduction = None, backend=None, checkpoint_dir=None, n_jobs=-1):
    if reduction is None:
        reduction=SaveMetricsToDisk(filename)
    exe = JobLibExec(
        seq=seq, 
        post=PostProcess(
            metrics=metrics,
            reduction=reduction,
        ),
        backend=backend,
        checkpoint_dir=checkpoint_dir
    )
    exe(n_jobs=n_jobs)
    

def run_dask(seq: SimSeq, metrics: List[Metric], client, filename='results',
             reduction: Reduction = None, backend=None, checkpoint_dir=None):
    if reduction is None:
        reduction=SaveMetricsToDisk(filename)
    exe = DaskExec(
        seq=seq, 
        post=PostProcess(
            metrics=metrics,
            reduction=reduction,
        ),
        backend=backend,
        checkpoint_dir=checkpoint_dir
    )
    exe(client)
