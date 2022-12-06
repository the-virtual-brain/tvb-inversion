from typing import Dict, Optional, Union, Callable, List

import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt
import arviz as az
import matplotlib.pyplot as plt

from tvb_inversion.base.inference import Estimator
from tvb_inversion.pymc3.stats_model import Pymc3Model


class EstimatorPYMC(Estimator):

    def __init__(
            self,
            stats_model: Pymc3Model
    ):

        super().__init__(stats_model)

        self.inference_data = None
        self.inference_summary = None

    @property
    def model(self):
        return self.stats_model.model

    def run_inference(self, draws: int, tune: int, cores: int, target_accept: float):
        with self.model:
            trace = pm.sample(draws=draws, tune=tune, cores=cores, target_accept=target_accept)
            posterior_predictive = pm.sample_posterior_predictive(trace=trace)
            # prior_predictive = pm.sample_prior_predictive(samples=1000)
            self.inference_data = az.from_pymc3(trace=trace, posterior_predictive=posterior_predictive)
            self.inference_summary = az.summary(self.inference_data)
