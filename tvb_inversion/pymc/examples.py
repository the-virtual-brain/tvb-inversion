from typing import Optional
import pymc as pm
import numpy as np
import pytensor
import pytensor.tensor as pyt

from tvb.simulator.lab import *
from tvb.simulator.backend.pytensor import PytensorBackend
from tvb_inversion.base.observation_models import linear
from tvb_inversion.pymc.prior import PymcPrior
from tvb_inversion.pymc.stats_model import PymcModel
from tvb_inversion.pymc.stats_model_builder import (StochasticPymcModelBuilder, DefaultStochasticPymcModelBuilder)
from tvb_inversion.pymc.inference import EstimatorPYMC

np.random.seed(42)


def create_2node_simulator(simulation_length: float):
    conn = connectivity.Connectivity()
    conn.weights = np.array([[0., 1.], [1., 0.]])
    conn.region_labels = np.array(["R1", "R2"])
    conn.centres = np.random.rand(2, 3)
    conn.tract_lengths = np.array([[0., 2.], [2., 0.]])
    conn.configure()

    sim = simulator.Simulator(
        model=models.oscillator.Generic2dOscillator(a=np.array([0.75, 2.25])),
        connectivity=conn,
        coupling=coupling.Linear(),
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


def create_10node_simulator(simulation_length: float):
    conn = connectivity.Connectivity()
    weights = np.random.normal(loc=1.0, scale=0.25, size=(10, 10))
    np.fill_diagonal(weights, 0.0)
    conn.weights = weights
    conn.region_labels = np.array([f"R{i}" for i in range(1, 11)])
    conn.centres = np.random.rand(10, 3)
    conn.tract_lengths = 2 * (np.ones((10, 10)) - np.eye(10))
    conn.configure()

    sim = simulator.Simulator(
        model=models.oscillator.Generic2dOscillator(a=np.random.normal(loc=1.5, scale=0.75, size=(10,))),
        connectivity=conn,
        coupling=coupling.Linear(),
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


def default_model_builders(
        sim: simulator.Simulator,
        observation: np.ndarray,
        save_file: Optional[str] = None,
        **sample_kwargs
):
    def_std = 0.1

    model = pm.Model()
    with model:
        model_a_star = pm.Normal(
            name="model_a_star", mu=0.0, sigma=1.0, shape=sim.model.a.shape)
        model_a = pm.Deterministic(
            name="model_a", var=sim.model.a * (1.0 + def_std * model_a_star))

        coupling_a_star = pm.Normal(
            name="coupling_a_star", mu=0.0, sigma=1.0)
        coupling_a = pm.Deterministic(
            name="coupling_a", var=sim.coupling.a[0].item() * (1.0 + def_std * coupling_a_star))

    prior = PymcPrior(
        model=model,
        names=["model.a", "coupling.a"],
        dist=[model_a, coupling_a]
    )

    model_builder = DefaultStochasticPymcModelBuilder(
        sim=sim, params=prior, observation=observation[:, :, :, 0])
    model_builder.set_default_prior(def_std=def_std)
    model_builder.compose_model()
    pymc_model = model_builder.build()
    pymc_estimator = EstimatorPYMC(stats_model=pymc_model)

    inference_data, inference_summary = pymc_estimator.run_inference(**sample_kwargs)

    if save_file is not None:
        inference_data.to_netcdf(filename=save_file + "_idata.nc", compress=False)
        inference_summary.to_json(path_or_buf=save_file + "_isummary.json")

    return inference_data, inference_summary


def uninformative_model_builders(
        sim: simulator.Simulator,
        observation: np.ndarray,
        save_file: Optional[str] = None,
        **sample_kwargs
):
    def_std = 0.1

    model = pm.Model()
    with model:
        model_a_star = pm.Normal(
            name="model_a_star", mu=0.0, sigma=1.0, shape=sim.model.a.shape)
        model_a = pm.Deterministic(
            name="model_a", var=sim.model.a * (1.0 + def_std * model_a_star))

        coupling_a_star = pm.Normal(
            name="coupling_a_star", mu=0.0, sigma=1.0)
        coupling_a = pm.Deterministic(
            name="coupling_a", var=sim.coupling.a[0].item() * (1.0 + def_std * coupling_a_star))

        x_init_star = pm.Normal(
            name="x_init_star", mu=0.0, sigma=1.0, shape=sim.initial_conditions.shape[:-1])
        x_init = pm.Deterministic(
            name="x_init", var=sim.initial_conditions[:, :, :, 0] * (1.0 + def_std * x_init_star))

        nsig_star = pm.Normal(
            name="nsig_star", mu=0.0, sigma=1.0)
        nsig = pm.Deterministic(
            name="nsig", var=sim.integrator.noise.nsig[0].item() * (1.0 + def_std * nsig_star))

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

    return inference_data, inference_summary


def custom_model_builders(
        sim: simulator.Simulator,
        observation: np.ndarray,
        save_file: Optional[str] = None,
        **sample_kwargs
):
    def_std = 0.1

    model = pm.Model()
    with model:
        model_a_star = pm.Normal(
            name="model_a_star", mu=0.0, sigma=1.0, shape=sim.model.a.shape)
        model_a = pm.Deterministic(
            name="model_a", var=sim.model.a * (1.0 + def_std * model_a_star))

        coupling_a_star = pm.Normal(
            name="coupling_a_star", mu=0.0, sigma=1.0)
        coupling_a = pm.Deterministic(
            name="coupling_a", var=sim.coupling.a[0].item() * (1.0 + def_std * coupling_a_star))

        x_init_star = pm.Normal(
            name="x_init_star", mu=0.0, sigma=1.0, shape=sim.initial_conditions.shape[:-1])
        x_init = pm.Deterministic(
            name="x_init", var=sim.initial_conditions[:, :, :, 0] * (1.0 + def_std * x_init_star))

        nsig_star = pm.Normal(
            name="nsig_star", mu=0.0, sigma=1.0)
        nsig = pm.Deterministic(
            name="nsig", var=sim.integrator.noise.nsig[0].item() * (1.0 + def_std * nsig_star))

        dWt_star = pm.Normal(
            name="dWt_star", mu=0.0, sigma=1.0, shape=(observation.shape[0], sim.model.nvar, sim.connectivity.number_of_regions))
        dWt = pm.Deterministic(
            name="dWt", var=pyt.sqrt(2.0 * nsig * sim.integrator.dt) * dWt_star)

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

    pymc_model = PymcModel(sim=sim, params=prior)

    def create_backend_funs(sim: simulator.Simulator):
        # Create pytensor backend functions
        template_dfun = """
                   import pytensor
                   import pytensor.tensor as pyt
                   import numpy as np
                   <%include file="pytensor-dfuns.py.mako"/>
                   """
        dfun = PytensorBackend().build_py_func(
            template_source=template_dfun, content=dict(sim=sim, mparams=list(prior.get_model_params().keys())), name="dfuns", print_source=True)

        template_cfun = f"""
                   import pytensor
                   import pytensor.tensor as pyt
                   import numpy as np
                   n_node = {sim.connectivity.number_of_regions}
                   <%include file="pytensor-coupling.py.mako"/>
                   """

        cfun = PytensorBackend().build_py_func(
            template_source=template_cfun, content=dict(sim=sim, cparams=list(prior.get_coupling_params().keys())), name="coupling", print_source=True)

        return dfun, cfun

    def scheme(dWt, x_prev, *params):

        state = pyt.zeros((sim.connectivity.horizon, sim.model.nvar, sim.connectivity.number_of_regions))
        state = pyt.set_subtensor(state[0], x_prev)
        state = pyt.transpose(state, axes=[1, 0, 2])

        cX = pyt.zeros((sim.history.n_cvar, sim.history.n_node))
        cX = cfun(cX, sim.connectivity.weights, state, sim.connectivity.delay_indices,
                  **prior.get_coupling_params())

        dX = pyt.zeros((sim.model.nvar, sim.history.n_node))
        dX = dfun(dX, x_prev, cX, sim.model.spatial_parameter_matrix, **prior.get_model_params())

        return x_prev + sim.integrator.dt * dX + dWt

    dfun, cfun = create_backend_funs(sim)
    with pymc_model.model:
        taps = list(-1 * np.arange(sim.connectivity.idelays.max() + 1) - 1)[::-1]
        x_sim, updates = pytensor.scan(
            fn=scheme,
            sequences=[dWt],
            outputs_info=[x_init[-1]],
            non_sequences=list(prior.get_model_params().values()) + list(prior.get_coupling_params().values()),
            n_steps=observation.shape[0]
        )

        x_hat = pm.Deterministic(
            name="x_hat", var=linear(x_sim, **prior.get_observation_model_params()))

        x_obs = pm.Normal(
            name="x_obs", mu=x_hat[:, sim.model.cvar, :], sigma=prior.dict.get("measurement_noise", 1.0), shape=observation.shape[:-1], observed=observation[:, :, :, 0])

    pymc_estimator = EstimatorPYMC(stats_model=pymc_model)

    inference_data, inference_summary = pymc_estimator.run_inference(**sample_kwargs)

    if save_file is not None:
        inference_data.to_netcdf(filename=save_file + "_idata.nc", compress=False)
        inference_summary.to_json(path_or_buf=save_file + "_isummary.json")

    return inference_data, inference_summary
