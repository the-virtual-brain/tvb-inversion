import os
import numpy
import sys

from tvb.core.neocom.h5 import load_ht

from tvb.simulator.backend.nb_mpr import NbMPRBackend

from tvb.config.init.datatypes_registry import populate_datatypes_registry


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


if __name__ == "__main__":
    simulator_gid = sys.argv[1]
    run_simulation(simulator_gid)
