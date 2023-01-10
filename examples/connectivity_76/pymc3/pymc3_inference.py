from typing import Optional
import os
import json
from datetime import datetime
import numpy as np
import pymc3 as pm

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


def create_simulator(simulation_length: float):
    conn = connectivity.Connectivity.from_file()

    sim = simulator.Simulator(
        model=models.oscillator.Generic2dOscillator(a=np.array([1.5])),
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
        "model_a": sim.model.a[0].item(),
        "coupling_a": sim.coupling.a[0].item(),
        "nsig": sim.integrator.noise.nsig[0].item()
    }

    model = pm.Model()
    with model:
        model_a_star = pm.Normal(
            name="model_a_star", mu=0.0, sd=1.0)
        model_a = pm.Deterministic(
            name="model_a", var=inference_params["model_a"] * (1.0 + def_std * model_a_star))

        coupling_a_star = pm.Normal(
            name="coupling_a_star", mu=0.0, sd=1.0)
        coupling_a = pm.Deterministic(
            name="coupling_a", var=inference_params["coupling_a"] * (1.0 + def_std * coupling_a_star))

        x_init_star = pm.Normal(
            name="x_init_star", mu=0.0, sd=1.0, shape=sim.initial_conditions.shape[:-1])
        x_init = pm.Deterministic(
            name="x_init", var=sim.initial_conditions[:, :, :, 0] * (1.0 + def_std * x_init_star))

        # BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
        # nsig_star = BoundedNormal(
        #     name="nsig_star", mu=0.0, sd=1.0)
        nsig_star = pm.Normal(
            name="nsig_star", mu=0.0, sd=1.0)
        nsig = pm.Deterministic(
            name="nsig", var=inference_params["nsig"] * (1.0 + def_std * nsig_star))

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

    inference_data, inference_summary = pymc_estimator.run_inference(**sample_kwargs)

    if save_file is not None:
        inference_data.to_netcdf(filename=save_file + "_idata.nc", compress=False)
        inference_summary.to_json(path_or_buf=save_file + "_isummary.json")
        with open(save_file + "_iparams.json", "w") as f:
            json.dump(inference_params, f)

    return inference_data, inference_summary


if __name__ == "__main__":
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M")

    sim = create_simulator(simulation_length=300)
    (t, X), = sim.run()
    np.save(f"{PATH}/pymc3_data/simulation_{run_id}.npy", X)
    simulation_params = {
        "model_a": sim.model.a[0].item(),
        "coupling_a": sim.coupling.a[0].item(),
        "nsig": sim.integrator.noise.nsig[0].item()
    }
    with open(f"{PATH}/pymc3_data/{run_id}_sim_params.json", "w") as f:
        json.dump(simulation_params, f)

    _ = build_model(sim=sim, observation=X, save_file=f"{PATH}/pymc3_data/{run_id}",
                    draws=250, tune=250, cores=4, target_accept=0.8, max_treedepth=20)
