from typing import Callable, List, Optional
import numpy as np
import itertools
import pandas as pd

from tvb_inversion.base.stats_model import StatisticalModel
from tvb_inversion.logger.builder import get_logger
from tvb_inversion.base.metrics import Metric


class Estimator:

    estimator: Callable

    def __init__(
            self,
            stats_model: StatisticalModel,
            observation: Optional[np.ndarray] = None,
            metrics: Optional[List[Metric]] = None
    ):

        self.logger = get_logger(self.__class__.__module__)

        self.stats_model = stats_model
        self.metrics = metrics
        self.obs = observation

    @property
    def prior(self):
        return self.stats_model.params

    @property
    def sim(self):
        return self.stats_model.sim

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


def zscore(true_mean, post_mean, post_std):
    """
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

    """
    return np.abs((post_mean - true_mean) / post_std)


def shrinkage(prior_std, post_std):
    """
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

    """
    return 1 - (post_std / prior_std)**2
