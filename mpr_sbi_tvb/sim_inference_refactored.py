import numpy as np
import sbi_tvb
from sbi_tvb.inference import TvbInference
from sbi_tvb.prior import Prior
from tvb.simulator.lab import *


def set_sim_params(_sim: simulator.Simulator, params):
    _sim.coupling.a = np.r_[params[0]]


def build_simulator():
    sbi_tvb_path = os.path.dirname(os.path.dirname(sbi_tvb.__file__))
    weights = np.loadtxt(os.path.join(sbi_tvb_path, 'data_input_files', 'SC_Schaefer7NW100p_nolog10.txt'))

    model = models.MontbrioPazoRoxin(
        eta=np.r_[-4.6],
        J=np.r_[14.5],
        Delta=np.r_[0.7],
        tau=np.r_[1],
    )

    magic_number = 124538.470647693
    weights_orig = weights / magic_number
    conn = connectivity.Connectivity(
        weights=weights_orig,
        region_labels=np.array(np.zeros(np.shape(weights_orig)[0]), dtype='<U128'),
        tract_lengths=np.zeros(np.shape(weights_orig)),
        areas=np.zeros(np.shape(weights_orig)[0]),
        speed=np.array(np.Inf, dtype=float),
        centres=np.zeros(np.shape(weights_orig)[0]))  # default 76 regions

    integrator = integrators.HeunStochastic(
        dt=0.005,
        noise=noise.Additive(
            nsig=np.r_[0.035, 0.035 * 2],
            noise_seed=42
        )
    )

    _monitors = [monitors.TemporalAverage(period=0.1)]

    cond_speed = np.Inf

    sim = simulator.Simulator(model=model,
                              connectivity=conn,
                              coupling=coupling.Scaling(
                                  a=np.r_[2.45]
                              ),
                              conduction_speed=cond_speed,
                              integrator=integrator,
                              monitors=_monitors,
                              simulation_length=30e3
                              )
    return sim


if __name__ == '__main__':
    sim = build_simulator()
    print("Build TvbInference object")
    tvb_inference = TvbInference(sim=sim,
                                 priors=[Prior('coupling.a', 1.5, 3.2),
                                         Prior('model.eta', -5, -1)])

    print("Sample priors")
    tvb_inference.sample_priors(num_simulations=10, num_workers=1)
    print("Train")
    tvb_inference.train()

    print("Run observed simulation")
    data = tvb_inference.run_sim([2.2])
    print("Run Posterior")
    post_samples = tvb_inference.posterior(data=data)
    print(post_samples.mean())
