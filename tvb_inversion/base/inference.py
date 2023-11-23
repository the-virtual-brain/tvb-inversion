from typing import Callable, List, Optional
import numpy as np
import itertools
import pandas as pd

from tvb_inversion.base.stats_model import StatisticalModel
# from tvb_inversion.logger.builder import get_logger
from tvb_inversion.base.metrics import Metric
from tvb_inversion.base.diagnostics import zscore, shrinkage


class Estimator:

    def __init__(
            self,
            stats_model: StatisticalModel,
            ground_truth: Optional[np.ndarray] = None
    ):
        self.logger = get_logger(self.__class__.__module__)
        self.stats_model = stats_model
        self.ground_truth = ground_truth

    @property
    def prior(self):
        return self.stats_model.params

    @property
    def sim(self):
        return self.stats_model.sim

    def compute_zscore(self, true_mean, posterior_mean, posterior_std):
        self.zscore = zscore(true_mean, posterior_mean, posterior_std)
        return self.zscore

    def compute_shrinkage(self, prior_std, posterior_std):
        self.shrinkage = shrinkage(prior_std, posterior_std)
        return self.shrinkage
