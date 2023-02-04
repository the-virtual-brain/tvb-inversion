from typing import Optional
import os
import json
from datetime import datetime
import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt

from tvb.simulator.lab import *
from tvb.simulator.backend.theano import TheanoBackend
from tvb_inversion.pymc3.inference import EstimatorPYMC
from tvb_inversion.pymc3.stats_model import Pymc3Model
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
    conn = connectivity.Connectivity()
    conn.weights = np.array([[1.]])
    conn.region_labels = np.array(["R1"])
    conn.centres = np.array([[0.1, 0.1, 0.1]])
    conn.tract_lengths = np.array([[0.]])
    conn.configure()

    sim = simulator.Simulator(
        model=models.oscillator.Generic2dOscillator(a=np.array([1.5])),
        connectivity=conn,
        coupling=coupling.Difference(),
        integrator=integrators.EulerStochastic(
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
        "model_a": sim.model.a[0] + 0.5 * sim.model.a[0],
        "nsig": sim.integrator.noise.nsig[0] + 0.5 * sim.integrator.noise.nsig[0]
    }

    model = pm.Model()
    with model:
        model_a_star = pm.Normal(
            name="model_a_star", mu=0.0, sd=1.0)
        # model_a = pm.Deterministic(
        #    name="model_a", var=inference_params["model_a"] * (1.0 + def_std * model_a_star))
        model_a = pm.Deterministic(
            name="model_a", var=inference_params["model_a"] + def_std * sim.model.a[0] * model_a_star)

        x_init_star = pm.Normal(
            name="x_init_star", mu=0.0, sd=1.0, shape=sim.initial_conditions.shape[1:-1])
        x_init = pm.Deterministic(
            name="x_init", var=sim.initial_conditions[0, :, :, 0] * (1.0 + def_std * x_init_star))

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
        dWt = pm.Deterministic(
            name="dWt", var=tt.sqrt(2.0 * nsig * sim.integrator.dt) * dWt_star)

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
        names=["model.a", "x_init", "integrator.noise.nsig", "dWt_star",
               "observation.amplitude", "observation.offset", "measurement_noise"],
        dist=[model_a, x_init, nsig, dWt_star,
              amplitude, offset, measurement_noise]
    )

    pymc_model = Pymc3Model(sim=sim, params=prior)

    def create_backend_funs(sim: simulator.Simulator):
        # Create theano backend functions
        template_dfun = """
                       import theano
                       import theano.tensor as tt
                       import numpy as np
                       <%include file="theano-dfuns.py.mako"/>
                       """
        dfun = TheanoBackend().build_py_func(
            template_source=template_dfun, content=dict(sim=sim, mparams=list(prior.get_model_params().keys())), name="dfuns", print_source=False)

        # template_cfun = f"""
        #                import theano
        #                import theano.tensor as tt
        #                import numpy as np
        #                n_node = {sim.connectivity.number_of_regions}
        #                <%include file="theano-coupling.py.mako"/>
        #                """
        #
        # cfun = TheanoBackend().build_py_func(
        #     template_source=template_cfun, content=dict(sim=sim), name="coupling", print_source=True)

        return dfun

    def scheme(dWt, x_prev):
        # x_prev = x_prev[::-1]

        # state = tt.stack(x_prev, axis=0)
        # state = tt.transpose(state, axes=[1, 0, 2])

        cX = tt.zeros((sim.history.n_cvar, sim.history.n_node))
        # cX = cfun(cX, sim.connectivity.weights, state, sim.connectivity.delay_indices,
        #           **prior.get_coupling_params())

        dX = tt.zeros((sim.model.nvar, sim.history.n_node))
        dX = dfun(dX, x_prev, cX, sim.model.spatial_parameter_matrix, **prior.get_model_params())

        return x_prev + sim.integrator.dt * dX + dWt

    dfun = create_backend_funs(sim)
    with pymc_model.model:
        # taps = list(-1 * np.arange(sim.connectivity.idelays.max() + 1) - 1)[::-1]
        x_sim, updates = theano.scan(
            fn=scheme,
            sequences=[dWt],
            outputs_info=[x_init],
            n_steps=observation.shape[0]
        )

        x_hat = pm.Deterministic(
            name="x_hat", var=linear(x_sim, **prior.get_observation_model_params()))

        x_obs = pm.Normal(
            name="x_obs", mu=x_hat[:, sim.model.cvar, :], sd=prior.dict.get("measurement_noise", 1.0), shape=observation.shape[:-1], observed=observation[:, :, :, 0])

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

    sim = create_simulator(simulation_length=250)
    (t, X), = sim.run()
    np.save(f"{PATH}/pymc3_data/simulation_{run_id}.npy", X)
    simulation_params = {
        "model_a": sim.model.a[0],
        "nsig": sim.integrator.noise.nsig[0]
    }
    with open(f"{PATH}/pymc3_data/{run_id}_sim_params.json", "w") as f:
        json.dump(simulation_params, f)

    _ = build_model(sim=sim, observation=X, save_file=f"{PATH}/pymc3_data/{run_id}",
                    draws=500, tune=500, cores=4, target_accept=0.95, max_treedepth=15)
