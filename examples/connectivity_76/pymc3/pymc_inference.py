import pymc3 as pm
import numpy as np
from datetime import datetime

from tvb.simulator.lab import *
from tvb_inversion.pymc3.prior import Pymc3Prior
from tvb_inversion.pymc3.stats_model import Pymc3Model
from tvb_inversion.pymc3.inference import EstimatorPYMC


run_id = datetime.now().strftime("%Y-%m-%d_%H%M")

conn = connectivity.Connectivity.from_file()
sim = simulator.Simulator(
    model=models.oscillator.Generic2dOscillator(),
    connectivity=conn,
    coupling=coupling.Difference(),
    integrator=integrators.HeunStochastic(
        dt=1.0,
        noise=noise.Additive(
            nsig=np.array([0.003]),
            noise_seed=42
        )
    ),
    monitors=[monitors.Raw()],
    simulation_length=250
)

sim.configure()
(t, X), = sim.run()
np.save(f"pymc3_data/sim_{run_id}.npy")

model = pm.Model()
with model:
    # a_model_star = pm.Normal(name="a_model_star", mu=0.0, sd=1.0)
    # a_model = pm.Deterministic(name="a_model", var=-2.0 + 1.0 * a_model_star)

    a_coupling_star = pm.Normal(name="a_coupling_star", mu=0.0, sd=1.0)
    a_coupling = pm.Deterministic(name="a_coupling", var=0.1 + 0.05 * a_coupling_star)

    BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
    noise_gfun_star = BoundedNormal(name="noise_gfun_star", mu=0.0, sd=1.0)
    noise_gfun = pm.Deterministic(name="noise_gfun", var=0.07 + 0.1 * noise_gfun_star)
    # noise_gfun = sim.integrator.noise.gfun(None)[0]

    noise_star = pm.Normal(name="noise_star", mu=0.0, sd=1.0, shape=X.shape[:-1])
    dynamic_noise = pm.Deterministic(name="dynamic_noise", var=noise_gfun * noise_star)

    global_noise = pm.HalfNormal(name="global_noise", sigma=0.05)

prior = Pymc3Prior(
    names=["coupling.a", "dynamic_noise", "global_noise"],
    dist=[a_coupling, dynamic_noise, global_noise]
)
pymc_model = Pymc3Model(sim=sim, params=prior, model=model)
pymc_estimator = EstimatorPYMC(stats_model=pymc_model, observation=X)

draws = 250
tune = 250
cores = 4

inference_data = pymc_estimator.run_inference(draws, tune, cores, target_accept=0.9)

pymc_estimator.inference_data.to_netcdf(filename=f"pymc3_data/{run_id}.nc", compress=False)
