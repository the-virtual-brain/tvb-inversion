import os
import tempfile
from copy import deepcopy
from enum import Enum
from typing import Callable, List, Optional
import numpy as np
import sbi.inference as sbi_inference
import sbi.inference
from sbi.utils import torchutils
import scipy
import torch
import itertools
import pandas as pd

from tvb_inversion.logger.builder import get_logger
from tvb_inversion.parameters import SimSeq, Metric
from .prior import Prior

class EstimatorSBI:
    def __init__(self, 
            prior: Prior, 
            seq: Optional[SimSeq] = None, 
            theta: Optional[torch.Tensor] = None,
            metrics: Optional[List[Metric]] = None, 
            method='SNPE'):

        self.logger = get_logger(self.__class__.__module__)


        self.prior = prior
        self.seq = seq
        self.theta = theta
        self.metrics = metrics
        self._method = method

        if theta is None:
            assert seq is not None, "Provide seq or theta."
            theta = np.array(self.seq.values).squeeze()
            theta = torch.as_tensor(theta[:,np.newaxis],dtype=torch.float32)
        elif seq is not None:
            self.logger.info(f'Ignoring provided seq.')

        try:
            method_fun: Callable = getattr(sbi_inference, method.upper())
        except AttributeError:
            raise NameError(
                "Method not available. `method` must be one of 'SNPE', 'SNLE', 'SNRE'."
            )
        self.estimator = method_fun(self.prior.dist)


    def load_summary_stats(self, filename: str):
        assert self.metrics is not None, 'Metrics not provided.'
        results = np.load(filename, allow_pickle=True)
        idx = list(itertools.chain.from_iterable([m.summary_stats_idx for m in self.metrics ]))
        columns = list(itertools.chain.from_iterable([m.summary_stats_labels for m in self.metrics ]))

        summary_stats = pd.DataFrame.from_records(
            results[:,idx].astype(np.float64), 
            columns=columns
        )

        return summary_stats



    def train(self, summary_stats: np.ndarray) -> "Posterior":
        x = torch.as_tensor(summary_stats,dtype=torch.float32)

        self.logger.info(f'Starting training with {self._method}...')
        _ = self.estimator.append_simulations(self.theta, x).train()
        posterior = self.estimator.build_posterior()
        self.logger.info(f'Finished training with {self._method}...')
        self.posterior = posterior
        return posterior


def zscore(true_mean, post_mean, post_std):
    '''
    calculate z-score

    parameters
    ------------
    true_mean: float
        true value of the parameter
    post_mean: float
        mean [max] value of the posterior
    post_std: float
        standard deviation of postorior

    return
    --------

    z-score: float

    '''
    return np.abs((post_mean - true_mean) / post_std)


def shrinkage(prior_std, post_std):
    '''
    shrinkage = 1 -  \frac{sigma_{post}/sigma_{prior}} ^2

    parameters
    -----------
    prior_std: float
        standard deviation of prior
    post_std: float
        standard deviation of postorior

    return
    ----------
    shrinkage: float

    '''
    return 1 - (post_std / prior_std)**2
