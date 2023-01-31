import os
import json
import numpy as np
import torch
from tqdm import tqdm
from joblib import Parallel, delayed
from datetime import datetime

from sbi.utils.user_input_checks import process_prior
from tvb.simulator.lab import *

import tvb_inversion.base.sim_seq
from tvb_inversion.sbi.prior import PytorchPrior
from tvb_inversion.sbi.stats_model import SBIModel
from tvb_inversion.sbi import EstimatorSBI

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


def run_seq(sim_seq: tvb_inversion.base.sim_seq.SimSeq):
    theta = np.array(sim_seq.values).squeeze()
    pool = Parallel(16)

    @delayed
    def job(i, sim_):
        (t, y), = sim_.configure().run()

        eps = theta[i, -1]
        # y_obs = y.flatten(order="F") + np.random.multivariate_normal(mean=np.zeros(y.size), cov=np.diag(eps * np.ones(y.size)))
        y_obs = y.flatten()
        return y_obs

    results = pool(job(i, sim_) for i, sim_ in tqdm(enumerate(sim_seq)))

    return results


if __name__ == "__main__":
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M")

    sim = create_simulator(simulation_length=250)
    (_, X), = sim.run()
    np.save(f"{PATH}/sbi_data/simulation_{run_id}.npy", X)

    sim_params = {
        "model_a": sim.model.a[0],
        "nsig": sim.integrator.noise.nsig[0],
        # "measurement_noise": 0.0
    }

    def_std = 0.5
    inference_params = {
        "model_a": sim.model.a[0],  # + 0.5 * sim.model.a[0],
        "nsig": sim.integrator.noise.nsig[0],  # + 0.5 * sim.integrator.noise.nsig[0],
        # "measurement_noise": 0.0
    }
    # loc = np.log(inference_params["nsig"] ** 2 / np.sqrt(inference_params["nsig"] ** 2 + (def_std * sim.integrator.noise.nsig[0]) ** 2))
    # scale = np.log(1 + (def_std * sim.integrator.noise.nsig[0]) ** 2 / inference_params["nsig"] ** 2)
    loc = np.log(inference_params["nsig"] ** 2 / np.sqrt(inference_params["nsig"] ** 2 + (def_std * sim.integrator.noise.nsig[0]) ** 2))
    scale = np.log(1 + (def_std * sim.integrator.noise.nsig[0]) ** 2 / inference_params["nsig"] ** 2)

    param_names = ["model.a", "integrator.noise.nsig"]  # , "measurement_noise"]
    param_dists = [
        torch.distributions.Normal(
            loc=torch.Tensor([inference_params["model_a"]]),
            scale=torch.Tensor([def_std * sim.model.a[0]])
        ),
        torch.distributions.LogNormal(
            loc=torch.Tensor([loc]),
            scale=torch.Tensor([scale])
        ),
        # torch.distributions.HalfNormal(torch.Tensor([0.1]))
    ]
    dist, _, _ = process_prior(param_dists)

    prior = PytorchPrior(param_names, dist)
    sbi_model = SBIModel(sim, prior)
    seq = sbi_model.generate_sim_seq(2000)
    estimator = EstimatorSBI(stats_model=sbi_model, seq=seq)

    simulations = run_seq(sim_seq=seq)
    simulations = np.asarray(simulations, dtype=np.float32)
    # simulations = simulations.reshape((simulations.shape[0], simulations[0].size), order="F")

    len_train_data = 1600
    posterior = estimator.train(simulations, len_train_data)
    posterior_samples = []
    for x in simulations[len_train_data:]:
        posterior_samples_ = posterior.sample((2000, ), x)
        posterior_samples.append(np.asarray(posterior_samples_))
    posterior_samples = np.asarray(posterior_samples)

    np.save(f"{PATH}/sbi_data/training_sims_{run_id}.npy", np.asarray(simulations[:len_train_data]))
    np.save(f"{PATH}/sbi_data/test_sims_{run_id}.npy", np.asarray(simulations[len_train_data:]))
    np.save(f"{PATH}/sbi_data/prior_samples_{run_id}.npy", np.asarray(estimator.theta))
    np.save(f"{PATH}/sbi_data/posterior_samples_{run_id}.npy", np.asarray(posterior_samples))
    with open(f"{PATH}/sbi_data/sim_params_{run_id}.json", "w") as f:
        json.dump(sim_params, f)
        f.close()
    with open(f"{PATH}/sbi_data/inference_params_{run_id}.json", "w") as f:
        json.dump(inference_params, f)
        f.close()
    with open(f"{PATH}/sbi_data/summary_{run_id}.json", "w") as f:
        json.dump(estimator.estimator._summary, f)
        f.close()
