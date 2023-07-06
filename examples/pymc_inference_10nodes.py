from typing import Optional
import os
import json
from datetime import datetime
import numpy as np
import pymc as pm

from tvb.simulator.lab import *
from tvb_inversion.pymc.inference import EstimatorPYMC
from tvb_inversion.pymc.prior import PymcPrior
from tvb_inversion.pymc.stats_model_builder import StochasticPymcModelBuilder
from tvb_inversion.base.observation_models import linear
from tvb_inversion.pymc.examples import create_10node_simulator

PATH = os.path.dirname(__file__)
np.random.seed(42)


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
            name="model_a_star", mu=0.0, sigma=1.0, shape=sim.model.a.shape)
        model_a = pm.Deterministic(
            name="model_a", var=inference_params["model_a"] + 0.75 * model_a_star)

        coupling_a_star = pm.Normal(
            name="coupling_a_star", mu=0.0, sigma=1.0)
        coupling_a = pm.Deterministic(
            name="coupling_a", var=inference_params["coupling_a"] + def_std * sim.coupling.a[0] * coupling_a_star)

        x_init_star = pm.Normal(
            name="x_init_star", mu=0.0, sigma=1.0, shape=sim.initial_conditions.shape[:-1])
        x_init = pm.Deterministic(
            name="x_init", var=sim.initial_conditions[:, :, :, 0] * (1.0 + def_std * x_init_star))

        nsig_star = pm.Normal(
            name="nsig_star", mu=0.0, sigma=1.0)
        nsig = pm.Deterministic(
            name="nsig", var=inference_params["nsig"] + def_std * sim.integrator.noise.nsig[0] * nsig_star)

        dWt_star = pm.Normal(
            name="dWt_star", mu=0.0, sigma=1.0, shape=(observation.shape[0], sim.model.nvar, sim.connectivity.number_of_regions))

        amplitude_star = pm.Normal(
            name="amplitude_star", mu=0.0, sigma=1.0)
        amplitude = pm.Deterministic(
            name="amplitude", var=1.0 * (1.0 + def_std * amplitude_star))

        offset_star = pm.Normal(
            name="offset_star", mu=0.0, sigma=1.0)
        offset = pm.Deterministic(
            name="offset", var=def_std * offset_star)

        observation_noise_star = pm.HalfNormal(
            name="observation_noise_star", sigma=1.0)
        observation_noise = pm.Deterministic(
            name="observation_noise", var=def_std * observation_noise_star)

    prior = PymcPrior(
        model=model,
        names=["model.a", "coupling.a", "x_init", "integrator.noise.nsig", "dWt_star",
               "observation.amplitude", "observation.offset", "observation_noise"],
        dist=[model_a, coupling_a, x_init, nsig, dWt_star,
              amplitude, offset, observation_noise]
    )

    model_builder = StochasticPymcModelBuilder(
        sim=sim, params=prior, observation_fun=linear, observation=observation[:, :, :, 0])
    model_builder.compose_model()
    pymc_model = model_builder.build()
    pymc_estimator = EstimatorPYMC(stats_model=pymc_model)

    inference_data, inference_summary = pymc_estimator.run_inference(**sample_kwargs)

    if save_file is not None:
        inference_data.to_netcdf(filename=save_file + "_idata.nc", compress=False)
        inference_summary.to_json(path_or_buf=save_file + "_isummary.json")
        with open(save_file + "_iparams.json", "w") as f:
            inference_params["model_a"] = inference_params["model_a"].tolist()
            json.dump(inference_params, f)

    return inference_data, inference_summary


if __name__ == "__main__":
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M")

    sim = create_10node_simulator(simulation_length=250)
    (t, X), = sim.run()
    np.save(f"{PATH}/pymc_data/simulation_{run_id}.npy", X)
    simulation_params = {
        "model_a": sim.model.a.tolist(),
        "coupling_a": sim.coupling.a[0],
        "nsig": sim.integrator.noise.nsig[0]
    }
    with open(f"{PATH}/pymc3_data/{run_id}_sim_params.json", "w") as f:
        json.dump(simulation_params, f)

    _ = build_model(sim=sim, observation=X, save_file=f"{PATH}/pymc3_data/{run_id}",
                    draws=500, tune=500, cores=4, target_accept=0.95)