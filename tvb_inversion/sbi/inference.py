import os
import tempfile
from copy import deepcopy
from enum import Enum
from typing import Callable, List
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
    def __init__(self, prior: Prior, seq: SimSeq, metrics: List[Metric], method='SNPE'):
        self.prior = prior
        self.seq = seq
        self.metrics = metrics
        self._method = method
        try:
            method_fun: Callable = getattr(sbi_inference, method.upper())
        except AttributeError:
            raise NameError(
                "Method not available. `method` must be one of 'SNPE', 'SNLE', 'SNRE'."
            )
        self.estimator = method_fun(self.prior.dist)

        self.logger = get_logger(self.__class__.__module__)

    def load_summary_stats(self, filename: str):
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
        theta = np.array(self.seq.values).squeeze()
        theta = torch.as_tensor(theta[:,np.newaxis],dtype=torch.float32)

        self.logger.info(f'Starting training with {self._method}...')
        _ = self.estimator.append_simulations(theta, x).train()
        posterior = self.estimator.build_posterior()
        self.logger.info(f'Finished training with {self._method}...')
        self.posterior = posterior
        return posterior
