from typing import Dict, Optional, Union, Callable, List

import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt
import arviz as az
import matplotlib.pyplot as plt
from tvb.simulator import noise

from tvb_inversion.base.inference import Estimator
from tvb_inversion.pymc3.stats_model import Pymc3Model
from tvb_inversion.base.metrics import Metric
from tvb.simulator.backend.theano import TheanoBackend
import tvb.simulator.noise


class EstimatorPYMC(Estimator):

    def __init__(
            self,
            stats_model: Pymc3Model,
            observation: np.ndarray,
            metrics: Optional[List[Metric]] = None
    ):

        super().__init__(stats_model,observation, metrics)

        self.mparams = {n.split(".")[-1]: d for (n, d) in zip(self.prior.names, self.prior.dist) if "model" in n}
        self.cparams = {n.split(".")[-1]: d for (n, d) in zip(self.prior.names, self.prior.dist) if "coupling" in n}
        self.iparams = {n.split(".")[-1]: d for (n, d) in zip(self.prior.names, self.prior.dist) if "integrator" in n}

        self.dfun: Callable = self.build_dfun()
        self.cfun: Callable = self.build_cfun()

        self.inference_data = None
        self.inference_summary = None

    @property
    def model(self):
        return self.stats_model.model

    def build_dfun(self):

        template = """
        import theano
        import theano.tensor as tt
        import numpy as np
        <%include file="theano-dfuns.py.mako"/>
        """

        dfun = TheanoBackend().build_py_func(template_source=template, content=dict(sim=self.sim), name="dfuns", print_source=False)
        return dfun

    def build_cfun(self):

        template = f"""
        import theano
        import theano.tensor as tt
        import numpy as np
        n_node = {self.sim.connectivity.number_of_regions}
        <%include file="theano-coupling.py.mako"/>
        """

        cfun = TheanoBackend().build_py_func(template_source=template, content=dict(sim=self.sim), name="coupling", print_source=False)
        return cfun

    def build_integrator(self):
        pass

    def scheme(self, x_eta, *x_prev):

        x_prev = x_prev[::-1]

        state = tt.stack(x_prev, axis=0)
        state = tt.transpose(state, axes=[1, 0, 2])

        cX = tt.zeros((self.sim.history.n_cvar, self.sim.history.n_node))
        cX = self.cfun(cX, self.sim.connectivity.weights, state, self.sim.connectivity.delay_indices, **self.cparams)

        dX = tt.zeros((self.sim.model.nvar, self.sim.history.n_node))
        parmat = self.sim.model.spatial_parameter_matrix
        dX = self.dfun(dX, x_prev[0], cX, parmat, **self.mparams)

        x_next = x_prev[0] + self.sim.integrator.dt * dX + x_eta
        return x_next

    def run_inference(self, draws: int, tune: int, cores: int, target_accept: float):

        with self.model:
            x_init = tt.zeros(self.sim.history.buffer.shape[:-1])
            x_init = tt.set_subtensor(x_init[-1], self.obs[0, :, :, 0])
            x_init = tt.unbroadcast(x_init, 2)

            if isinstance(self.sim.integrator.noise, noise.Additive):
                gfun = pm.Deterministic(name="gfun", var=tt.sqrt(2.0 * self.iparams["nsig"]))
                noise_star = pm.Normal(name="noise_star", mu=0.0, sd=1.0, shape=self.obs.shape[:-1])
                dynamic_noise = pm.Deterministic(name="dynamic_noise", var=gfun * noise_star)
            else:
                raise NotImplementedError

            taps = list(-1 * np.arange(self.sim.connectivity.idelays.max() + 1) - 1)[::-1]
            x_sim, updates = theano.scan(
                fn=self.scheme,
                sequences=[dynamic_noise],
                outputs_info=[dict(initial=x_init, taps=taps)],
                n_steps=self.obs.shape[0]
            )

            amplitude_star = pm.Normal(name="amplitude_star", mu=0.0, sd=1.0)
            amplitude = pm.Deterministic(name="amplitude", var=0.0 + amplitude_star)

            offset_star = pm.Normal(name="offset_star", mu=0.0, sd=1.0)
            offset = pm.Deterministic(name="offset", var=0.0 + offset_star)

            x_hat = pm.Deterministic(name="x_hat", var=amplitude * x_sim + offset)
            observation_noise = pm.HalfNormal(name="observation_noise", sigma=0.05)
            x_obs = pm.Normal(name="x_obs", mu=x_hat, sd=observation_noise, shape=self.obs.shape[:-1], observed=self.obs[:, :, :, 0])

            trace = pm.sample(draws=draws, tune=tune, cores=cores, target_accept=target_accept)
            posterior_predictive = pm.sample_posterior_predictive(trace=trace)
            # prior_predictive = pm.sample_prior_predictive(samples=1000)
            self.inference_data = az.from_pymc3(trace=trace, posterior_predictive=posterior_predictive)
            self.inference_summary = az.summary(self.inference_data)

            return self.inference_data


def plot_posterior_samples(inference_data, init_params: Dict[str, float], save: bool = False):
    num_params = len(init_params)
    nrows = int(np.ceil(np.sqrt(num_params)))
    ncols = int(np.ceil(num_params / nrows))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 16))
    for ax in axes.reshape(-1):
        ax.set_axis_off()
    for i, (key, value) in enumerate(init_params.items()):

        posterior_ = inference_data.posterior[key].values.reshape((inference_data.posterior[key].values.size,))
        ax = axes.reshape(-1)[i]
        ax.set_axis_on()
        ax.hist(posterior_, bins=100, alpha=0.5)
        ax.axvline(init_params[key], color="r", label="simulation parameter")
        ax.axvline(posterior_.mean(), color="k", label="posterior mean")

        ax.set_title(key, fontsize=18)
        ax.tick_params(axis="both", labelsize=16)
    try:
        axes[0, 0].legend(fontsize=18)
    except IndexError:
        axes[0].legend(fontsize=18)
