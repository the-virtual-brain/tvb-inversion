import os
from datetime import datetime
import numpy as np

from tvb.simulator.lab import *
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


if __name__ == "__main__":

    run_id = datetime.now().strftime("%Y-%m-%d_%H%M")

    sim = create_simulator(simulation_length=250)
    (t, X), = sim.run()
    np.save(f"{PATH}/pymc3_data/simulation_{run_id}.npy", X)

    _ = default_model_builders(sim=sim, observation=X, save_file=f"{PATH}/pymc3_data/{run_id}.nc",
                               draws=250, tune=250, cores=2, target_accept=0.9, max_treedepth=15)
    # _ = uninformative_model_builders(sim=sim, observation=X, save_file=f"{PATH}/pymc3_data/{run_id}.nc",
    #                                  draws=250, tune=250, cores=2, target_accept=0.9, max_treedepth=15)
    # _ = custom_model_builders(sim=sim, observation=X, save_file=f"{PATH}/pymc3_data/{run_id}.nc",
    #                           draws=250, tune=250, cores=2, target_accept=0.9, max_treedepth=15)
