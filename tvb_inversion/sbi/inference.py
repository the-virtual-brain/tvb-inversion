from typing import Callable, List, Optional
import numpy as np
import sbi.inference as sbi_inference
import torch
import itertools
import pandas as pd

from tvb_inversion.base.inference import Estimator, zscore, shrinkage
from tvb_inversion.sbi.stats_model import SBIModel
from tvb_inversion.logger.builder import get_logger
from tvb_inversion.base.parameters import SimSeq, Metric


class EstimatorSBI(Estimator):

    def __init__(
            self,
            stats_model: SBIModel,
            seq: Optional[SimSeq] = None,
            theta: Optional[torch.Tensor] = None,
            metrics: Optional[List[Metric]] = None,
            num_samples: Optional[int] = 0,
            method='SNPE'
    ):

        super().__init__(stats_model, metrics)
        self.logger = get_logger(self.__class__.__module__)

        self.seq = seq
        self.theta = theta
        self.metrics = metrics
        self._method = method

        if self.theta is None:
            if self.seq is None:
                assert num_samples > 0, "Provide number of samples, seq or theta."
                self.seq = self.stats_model.generate_sim_seq(num_samples=num_samples)
            self.theta = np.array(self.seq.values).squeeze()
            self.theta = torch.as_tensor(self.theta[:, np.newaxis], dtype=torch.float32)

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
        idx = list(itertools.chain.from_iterable([m.summary_stats_idx for m in self.metrics]))
        columns = list(itertools.chain.from_iterable([m.summary_stats_labels for m in self.metrics]))

        summary_stats = pd.DataFrame.from_records(
            results[:, idx].astype(np.float64),
            columns=columns
        )

        return summary_stats

    def train(self, summary_stats: np.ndarray) -> "Posterior":
        x = torch.as_tensor(summary_stats, dtype=torch.float32)

        self.logger.info(f'Starting training with {self._method}...')
        _ = self.estimator.append_simulations(self.theta, x).train()
        posterior = self.estimator.build_posterior()
        self.logger.info(f'Finished training with {self._method}...')
        self.posterior = posterior
        return posterior
