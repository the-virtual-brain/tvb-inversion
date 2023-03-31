from typing import Optional
import os
import json
from datetime import datetime
import numpy as np
import pymc3 as pm
import arviz as az
import pickle
from pymc3.step_methods.hmc import quadpotential
from pymc3.backends.base import MultiTrace

from tvb.simulator.lab import *
from tvb_inversion.pymc3.inference import EstimatorPYMC
from tvb_inversion.pymc3.prior import Pymc3Prior
from tvb_inversion.pymc3.stats_model_builder import StochasticPymc3ModelBuilder
from tvb_inversion.base.observation_models import linear
from tvb_inversion.pymc3.examples import (
    default_model_builders,
    uninformative_model_builders,
    custom_model_builders
)

PATH = os.path.dirname(__file__)
np.random.seed(42)


def create_simulator(simulation_length: float):
    conn = connectivity.Connectivity.from_file()
    Nr = conn.weights.shape[0]
    conn.weights = conn.weights - conn.weights*np.eye(Nr)
    conn.configure()

    sim = simulator.Simulator(
        model=models.oscillator.Generic2dOscillator(a=np.random.normal(loc=1.5, scale=0.75, size=(Nr,))),
        connectivity=conn,
        coupling=coupling.Difference(),
        integrator=integrators.HeunStochastic(
            dt=1.0,
            noise=noise.Additive(
                nsig=np.array([1e-4]),
                noise_seed=42
            )
        ),
        monitors=[monitors.Raw()],
        simulation_length=simulation_length
    )

    sim.configure()
    sim.initial_conditions = np.zeros((conn.horizon, sim.model.nvar, conn.number_of_regions, 1))
    sim.configure()

    return sim


def build_model(
        sim: simulator.Simulator,
        observation: np.ndarray,
        save_file: Optional[str] = None,
        **sample_kwargs
):
    def_std = 0.5
    inference_params = {
        "model_a": sim.model.a,
        # "model_a": 1.5 * np.ones(sim.model.a.shape),
        "coupling_a": sim.coupling.a[0],  # + 0.5 * sim.coupling.a[0],
        "nsig": sim.integrator.noise.nsig[0],  # + 0.5 * sim.integrator.noise.nsig[0]
    }

    model = pm.Model()
    with model:
        model_a_star = pm.Normal(
            name="model_a_star", mu=0.0, sd=1.0, shape=sim.model.a.shape)
        # model_a = pm.Deterministic(
        #     name="model_a", var=inference_params["model_a"] * (1.0 + def_std * model_a_star))
        model_a = pm.Deterministic(
            name="model_a", var=inference_params["model_a"] + 0.75 * model_a_star)

        coupling_a_star = pm.Normal(
            name="coupling_a_star", mu=0.0, sd=1.0)
        # coupling_a = pm.Deterministic(
        #    name="coupling_a", var=inference_params["coupling_a"] * (1.0 + def_std * coupling_a_star))
        coupling_a = pm.Deterministic(
            name="coupling_a", var=inference_params["coupling_a"] + def_std * sim.coupling.a[0] * coupling_a_star)

        x_init_star = pm.Normal(
            name="x_init_star", mu=0.0, sd=1.0, shape=sim.initial_conditions.shape[:-1])
        x_init = pm.Deterministic(
            name="x_init", var=sim.initial_conditions[:, :, :, 0] * (1.0 + def_std * x_init_star))

        # BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
        # nsig_star = BoundedNormal(
        #     name="nsig_star", mu=0.0, sd=1.0)
        nsig_star = pm.Normal(
            name="nsig_star", mu=0.0, sd=1.0)
        # nsig = pm.Deterministic(
        #    name="nsig", var=inference_params["nsig"] * (1.0 + def_std * nsig_star))
        nsig = pm.Deterministic(
            name="nsig", var=inference_params["nsig"] + def_std * sim.integrator.noise.nsig[0] * nsig_star)

        dWt_star = pm.Normal(
            name="dWt_star", mu=0.0, sd=1.0, shape=(observation.shape[0], sim.model.nvar, sim.connectivity.number_of_regions))

        amplitude_star = pm.Normal(
            name="amplitude_star", mu=0.0, sd=1.0)
        amplitude = pm.Deterministic(
            name="amplitude", var=1.0 * (1.0 + def_std * amplitude_star))

        offset_star = pm.Normal(
            name="offset_star", mu=0.0, sd=1.0)
        offset = pm.Deterministic(
            name="offset", var=def_std * offset_star)

        measurement_noise_star = pm.HalfNormal(
            name="measurement_noise_star", sd=1.0)
        measurement_noise = pm.Deterministic(
            name="measurement_noise", var=def_std * measurement_noise_star)

    prior = Pymc3Prior(
        model=model,
        names=["model.a", "coupling.a", "x_init", "integrator.noise.nsig", "dWt_star",
               "observation.amplitude", "observation.offset", "measurement_noise"],
        dist=[model_a, coupling_a, x_init, nsig, dWt_star,
              amplitude, offset, measurement_noise]
    )

    model_builder = StochasticPymc3ModelBuilder(
        sim=sim, params=prior, observation_fun=linear, observation=observation[:, :, :, 0])
    model_builder.compose_model()
    pymc_model = model_builder.build()
    pymc_estimator = EstimatorPYMC(stats_model=pymc_model)

    # inference_data, inference_summary = pymc_estimator.run_inference(**sample_kwargs)
    with pymc_estimator.model:
        trace1 = pm.load_trace(".pymc_4.trace")
        trace2 = pm.sample(trace=trace1, **sample_kwargs)
        posterior_predictive = pm.sample_posterior_predictive(trace=trace2)
        inference_data = az.from_pymc3(trace=trace2, posterior_predictive=posterior_predictive, save_warmup=True)
        inference_summary = az.summary(inference_data)

    if save_file is not None:
        inference_data.to_netcdf(filename=save_file + "_idata.nc", compress=False)
        inference_summary.to_json(path_or_buf=save_file + "_isummary.json")
        with open(save_file + "_iparams.json", "w") as f:
            inference_params["model_a"] = inference_params["model_a"].tolist()
            json.dump(inference_params, f)

    return trace


if __name__ == "__main__":
    run_id = "2023-03-13_1329"

    with open(f"{PATH}/pymc3_data/{run_id}_tuning_model.pkl", "rb") as buff:
        data = pickle.load(buff)

    model = data["model"]

    with model:
        # trace = data["trace"]
        trace = pm.load_trace(".pymc_1.trace/")

        chain1 = trace._straces[0][2:].samples
        chain2 = trace._straces[1][2:].samples
        chain3 = trace._straces[2][2:].samples
        # chain4 = trace._straces[3][:].samples

        # cov_est = pm.trace_cov(trace)
        rtrace = MultiTrace([trace._straces[0][2:], trace._straces[1][2:], trace._straces[2][2:]])
        cov_est = pm.trace_cov(rtrace)
        n = cov_est.shape[0]

        # mean = model.dict_to_array(trace[-25])
        # mean = np.asarray([model.dict_to_array(trace[i]) for i in range(len(trace))]).mean(axis=0)
        diverging1 = trace._straces[0]._get_sampler_stats(varname="diverging", sampler_idx=0, burn=0, thin=1)
        mean1 = np.asarray([model.dict_to_array(trace._straces[0][i]) for i in np.where(diverging1==False)[0]]).mean(axis=0)

        diverging2 = trace._straces[1]._get_sampler_stats(varname="diverging", sampler_idx=0, burn=0, thin=1)
        mean2 = np.asarray([model.dict_to_array(trace._straces[1][i]) for i in np.where(diverging2==False)[0]]).mean(axis=0)

        diverging3 = trace._straces[2]._get_sampler_stats(varname="diverging", sampler_idx=0, burn=0, thin=1)
        mean3 = np.asarray([model.dict_to_array(trace._straces[2][i]) for i in np.where(diverging3==False)[0]]).mean(axis=0)

        # diverging4 = trace._straces[3]._get_sampler_stats(varname="diverging", sampler_idx=0, burn=0, thin=1)
        # mean4 = np.asarray([model.dict_to_array(trace._straces[3][i]) for i in np.where(diverging4==False)[0]]).mean(axis=0)
        # mean4 = np.zeros(n)

        mean = np.vstack((mean1, mean2, mean3)).mean(axis=0)
        var = np.diag(cov_est)
        potential = quadpotential.QuadPotentialDiagAdapt(n, mean, var, 1)

        step_size = trace.get_sampler_stats("step_size")[np.where(trace.get_sampler_stats("diverging")==False)[0].tolist()].mean()
        step_scale = step_size  * (n ** 0.25)

        # start = [trace._straces[i][-1] for i in trace.chains]
        start1 = {var: val.mean(axis=0) for var, val in chain1.items()}
        start2 = {var: val.mean(axis=0) for var, val in chain2.items()}
        start3 = {var: val.mean(axis=0) for var, val in chain3.items()}
        # start4 = {var: val.mean(axis=0) for var, val in chain4.items()}
        start = [start1, start2, start3]

        nuts = pm.NUTS(
            potential=potential, target_accept=0.95, step_scale=step_scale, max_treedepth=10, adapt_step_size=True)
        nuts.tune = True

        trace2 = pm.sample(
            step=nuts, draws=500, tune=50, cores=3, start=start, discard_tuned_samples=False)
        # _ = pm.save_trace(trace2)

        posterior_predictive = pm.sample_posterior_predictive(trace=trace2)
        inference_data = az.from_pymc3(trace=trace2, posterior_predictive=posterior_predictive, save_warmup=True)
        inference_summary = az.summary(inference_data)

        inference_data.to_netcdf(filename=f"{PATH}/pymc3_data/{run_id}-3_idata.nc", compress=False)
        inference_summary.to_json(path_or_buf=f"{PATH}/pymc3_data/{run_id}-3_isummary.json")

    # run_id = datetime.now().strftime("%Y-%m-%d_%H%M")
    # run_id_load = "2023-02-28_1012"
    #
    # sim = create_simulator(simulation_length=250)
    # X = np.load(f"{PATH}/pymc3_data/simulation_{run_id_load}.npy")
    # np.save(f"{PATH}/pymc3_data/simulation_{run_id}.npy", X)
    # with open(f"pymc3_data/{run_id_load}_sim_params.json", "r") as f:
    #     simulation_params = json.load(f)
    #     f.close()
    # sim.model.a = np.array(simulation_params["model_a"])
    # print(sim.model.a)
    #
    # with open(f"{PATH}/pymc3_data/{run_id}_sim_params.json", "w") as f:
    #     json.dump(simulation_params, f)
    #
    # _ = build_model(sim=sim, observation=X, save_file=f"{PATH}/pymc3_data/{run_id}",
    #                 draws=100, tune=0, cores=2, target_accept=0.95, max_treedepth=10, discard_tuned_samples=False)
