import os
import time
import numpy as np
import sbi_tvb
from sbi_tvb.inference import TvbInference
from sbi_tvb.prior import Prior
from tvb.simulator.lab import *

LOGGER = get_logger(__name__)


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
                              coupling=coupling.Linear(
                                  a=np.r_[2.45]
                              ),
                              conduction_speed=cond_speed,
                              integrator=integrator,
                              monitors=_monitors,
                              simulation_length=30e3
                              )
    return sim


if __name__ == '__main__':
    os.environ['CLB_AUTH'] = ''
    sim = build_simulator()

    LOGGER.info("Build TvbInference object")
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'results')
    tvb_inference = TvbInference(sim=sim,
                                 priors=[Prior('coupling.a', 1.5, 3.2)],
                                 output_dir=output_dir)

    LOGGER.info("Sample priors")
    start = time.time()

    tvb_inference.sample_priors_remote(num_simulations=10, num_workers=1)

    end = time.time()
    LOGGER.info("Time elapsed for sim:", end - start)

    LOGGER.info("Train")
    tvb_inference.train()

    LOGGER.info("Run observed simulation")
    data = tvb_inference.run_sim(2.2 * np.ones(1))
    LOGGER.info("Run Posterior")
    post_samples = tvb_inference.posterior(data=data)
    LOGGER.info(f'Post samples mean value is {post_samples.mean()}')

    end = time.time()
    LOGGER.info("Time elapsed for all:", end - start)

    posterior = np.load(os.path.join(tvb_inference.output_dir, tvb_inference.POSTERIOR_SAMPLES))
    LOGGER.info(f'Posterior shape is {posterior.shape} and mean value is {posterior.mean()}')
