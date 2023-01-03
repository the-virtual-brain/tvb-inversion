import numpy as np
import torch
from tqdm import tqdm
from joblib import Parallel, delayed

from sbi.utils.user_input_checks import prepare_for_sbi, process_prior
from tvb.simulator.lab import *
from tvb_inversion.sbi.prior import PytorchPrior
from tvb_inversion.sbi.stats_model import SBIModel
from tvb_inversion.sbi import EstimatorSBI

simulation_length = 250

conn = connectivity.Connectivity()
conn.weights = np.array([[0., 2.], [2., 0.]])
conn.region_labels = np.array(["R1", "R2"])
conn.centres = np.array([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1]])
conn.tract_lengths = np.array([[0., 2.5], [2.5, 0.]])
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

names = ["model.a", "coupling.a", "integrator.noise.nsig"]
dists = [
    torch.distributions.Normal(torch.Tensor([1.5]), torch.Tensor([0.5])),
    torch.distributions.Normal(torch.Tensor([0.1]), torch.Tensor([0.05])),
    torch.distributions.HalfNormal(torch.Tensor([1e-4]), torch.Tensor([5e-5]))
]
dist, _, _ = process_prior(dists)

prior = PytorchPrior(names, dist)
sbi_model = SBIModel(sim, prior)
seq = sbi_model.generate_sim_seq(100)

pool = Parallel(4)


@delayed
def job(sim, i):
    (t, y), = sim.configure().run()
    return y


results = pool(job(_, i) for i, _ in tqdm(enumerate(seq)))

estimator = EstimatorSBI(stats_model=sbi_model, seq=seq)
x = np.asarray(results, dtype=np.float32)
x = x.reshape(x.shape[0], x[0].size)
estimator.train(x)
