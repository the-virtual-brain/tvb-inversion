import os
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


def run_seq(sim_seq: tvb_inversion.base.sim_seq.SimSeq):
    pool = Parallel(4)

    @delayed
    def job(sim_, i):
        (t, y), = sim_.configure().run()
        return y

    results = pool(job(sim_, i) for i, sim_ in tqdm(enumerate(sim_seq)))

    return results


if __name__ == "__main__":
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M")

    sim = create_simulator(simulation_length=250)
    (_, X), = sim.run()
    np.save(f"{PATH}/sbi_data/simulation_{run_id}.npy", X)

    param_names = ["model.a", "coupling.a", "integrator.noise.nsig"]
    param_dists = [
        torch.distributions.Normal(torch.Tensor([1.5]), torch.Tensor([0.5])),
        torch.distributions.Normal(torch.Tensor([0.1]), torch.Tensor([0.1])),
        torch.distributions.HalfNormal(torch.Tensor([1e-4]), torch.Tensor([5e-5]))
    ]
    dist, _, _ = process_prior(param_dists)

    prior = PytorchPrior(param_names, dist)
    sbi_model = SBIModel(sim, prior)
    seq = sbi_model.generate_sim_seq(20000)

    simulations = run_seq(sim_seq=seq)
    simulations = np.asarray(simulations, dtype=np.float32)
    simulations = simulations.reshape((simulations.shape[0], simulations[0].size), order="F")

    estimator = EstimatorSBI(stats_model=sbi_model, seq=seq)
    posterior = estimator.train(simulations)
    posterior_samples = posterior.sample((20000, ), torch.as_tensor(X.reshape(X.size, order="F")))

    np.save(f"{PATH}/sbi_data/training_sims_{run_id}.npy", np.asarray(simulations))
    np.save(f"{PATH}/sbi_data/prior_samples_{run_id}.npy", np.asarray(estimator.theta))
    np.save(f"{PATH}/sbi_data/posterior_samples_{run_id}.npy", np.asarray(posterior_samples))
