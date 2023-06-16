from typing import Dict, Optional, Union, Callable, List

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pyt
import arviz as az
import matplotlib.pyplot as plt

from tvb_inversion.base.inference import Estimator
from tvb_inversion.pymc.stats_model import PymcModel


class EstimatorPYMC(Estimator):

    def __init__(
            self,
            stats_model: PymcModel
    ):

        super().__init__(stats_model)

        self.trace = None
        self.inference_data = None
        self.posterior_predictive = None
        self.inference_summary = None

    @property
    def model(self):
        return self.stats_model.model

    def sample(self, **kwargs):
        with self.model:
            inference_data = pm.sample(**kwargs)
        self.inference_data = inference_data
        return self.inference_data

    def sample_posterior_predictive(self):
        with self.model:
            self.inference_data = pm.sample_posterior_predictive(trace=self.inference_data, extend_inferencedata=True)
        return self.inference_data

    def get_inference_data(self):
        with self.model:
            inference_data = az.from_pymc3(trace=self.trace, posterior_predictive=self.posterior_predictive)
        self.inference_data = inference_data
        return self.inference_data

    def get_inference_summary(self):
        self.inference_summary = az.summary(self.inference_data)
        return self.inference_summary

    def get_prior_std(self):
        raise NotImplementedError

    def get_posterior_mean(self, params: List[str]):
        posterior = np.concatenate(
            [self.inference_data.posterior[param].values.reshape(
                -1, self.inference_data.posterior[param].values.shape[-1]) if self.inference_data.posterior[param].values.ndim == 3 else self.inference_data.posterior[param].values.flatten()[..., np.newaxis] for param in
             params], axis=1)
        return posterior.mean(axis=0)

    def get_posterior_std(self, params: List[str]):
        posterior = np.concatenate(
            [self.inference_data.posterior[param].values.reshape(
                -1, self.inference_data.posterior[param].values.shape[-1]) if self.inference_data.posterior[param].values.ndim == 3 else self.inference_data.posterior[param].values.flatten()[..., np.newaxis] for param in
             params], axis=1)
        return posterior.std(axis=0)

    def information_criteria(self):
        waic = az.waic(self.inference_data)
        loo = az.loo(self.inference_data)
        return waic, loo

    def run_inference(self, **sample_kwargs):
        self.inference_data = self.sample(**sample_kwargs)
        self.inference_data = self.sample_posterior_predictive()
        self.inference_summary = self.get_inference_summary()
        return self.inference_data, self.inference_summary
