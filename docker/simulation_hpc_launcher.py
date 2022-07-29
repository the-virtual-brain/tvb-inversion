import os
import numpy
import sys

from tvb.core.neocom.h5 import load_ht
from tvb.simulator.backend.nb_mpr import NbMPRBackend
from tvb.config.init.datatypes_registry import populate_datatypes_registry

from sbi_tvb.inference import TvbInference
from sbi_tvb.logger.builder import get_logger
from sbi_tvb.prior import Prior

LOGGER = get_logger(__name__)


def run_simulation(simulator_gid):
    data_folder = "/home/data"

    populate_datatypes_registry()

    simulator = load_ht(simulator_gid, data_folder)
    conn = load_ht(simulator.connectivity.gid, data_folder)
    conn.configure()
    simulator.connectivity = conn
    simulator.configure()
    (temporal_average_time, temporal_average_data), = NbMPRBackend().run_sim(simulator,
                                                                             simulation_length=simulator.simulation_length)

    mysavepath = os.path.join(data_folder, 'time_series.npz')
    numpy.savez(mysavepath, data=temporal_average_data, time=temporal_average_time)


def sample_priors(simulator_gid, num_simulations, num_workers):
    LOGGER.info("Load simulator...")
    data_folder = "/home/data"

    populate_datatypes_registry()

    simulator = load_ht(simulator_gid, data_folder)
    conn = load_ht(simulator.connectivity.gid, data_folder)
    conn.configure()
    simulator.connectivity = conn
    simulator.configure()

    LOGGER.info("Build TvbInference object")
    tvb_inference = TvbInference(sim=simulator,
                                 priors=[Prior('coupling.a', 1.5, 3.2)])

    LOGGER.info("Sample priors")
    tvb_inference.sample_priors(num_simulations=num_simulations, num_workers=num_workers, save_path=data_folder)


if __name__ == "__main__":
    simulator_gid = sys.argv[1]
    num_simulations = int(sys.argv[2])
    num_workers = int(sys.argv[3])

    # run_simulation(simulator_gid)
    sample_priors(simulator_gid, num_simulations, num_workers)
