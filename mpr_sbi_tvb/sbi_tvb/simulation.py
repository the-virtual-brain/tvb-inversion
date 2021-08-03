import numpy as np
import itertools
from tvb.simulator.lab import *
from tvb.datatypes.time_series import TimeSeriesRegion
from tvb.analyzers import fmri_balloon
from sbi_tvb.bold import BalloonModel



def run_nbMPR_backend(sim, **kwargs):
    """Run networked Montbrio simulation using Numba backend.

    Parameters
    ----------
    sim : tvb.simulator
        Configured simulator instance. Should have single monitor (Raw or
        TemporalAverage), no regional vairance in parameters, no stimulus. 
    nstep : int
        number of iteration steps to perform. Mutually exclusive with simulation_length. 
    simulation_length : float
        Time of simulation in ms. Mutually exclusive with nstep.
    """
    from tvb.simulator.backend.nb_mpr import NbMPRBackend

    backend = NbMPRBackend()
    return backend.run_sim(sim, **kwargs)


def tavg_to_bold(tavg_t, tavg_d, sim=None, tavg_period=None, connectivity=None, svar=0, decimate=2000):
    if sim is not None:
        assert len(sim.monitors) == 1
        tavg_period = sim.monitors[0].period
        connectivity = sim.connectivity
    else:
        assert tavg_period is not None and connectivity is not None

    tsr = TimeSeriesRegion(
        connectivity = connectivity,
        data = tavg_d[:,[svar],:,:],
        time = tavg_t,
        sample_period = tavg_period
    )
    tsr.configure()

    bold_model = BalloonModel(time_series = tsr, dt=tavg_period/1000)
    bold_data_analyzer  = bold_model.evaluate()

    bold_t = bold_data_analyzer.time[::decimate] * 1000 # to ms
    bold_d = bold_data_analyzer.data[::decimate,:]

    return bold_t, bold_d 
