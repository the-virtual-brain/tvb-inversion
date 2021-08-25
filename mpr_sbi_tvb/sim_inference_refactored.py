import numpy as np
import sbi_tvb
from matplotlib import rcParams
from sbi_tvb.inference import TvbInference
from tvb.simulator.lab import *

rcParams['figure.figsize'] = 15, 6

if __name__ == '__main__':
    #
    # Simulation setup
    #
    print("Simulation setup")
    dt = 0.005
    nsigma = 0.035
    seed = 42
    sim_len = 30e3
    G = 2.45
    BOLD_TR = 2250

    sbi_tvb_path = os.path.dirname(os.path.dirname(sbi_tvb.__file__))
    weights = np.loadtxt(os.path.join(sbi_tvb_path, 'data_input_files', 'SC_Schaefer7NW100p_nolog10.txt'))

    mpr = models.MontbrioPazoRoxin(
        eta=np.r_[-4.6],
        J=np.r_[14.5],
        Delta=np.r_[0.7],
        tau=np.r_[1],
    )

    print("Build TvbInference object")
    tvb_inference = TvbInference('results', num_simulations=20, num_workers=6)
    print("Build prior")
    tvb_inference.build_prior(1.5, 3.2)
    print("Simulation setup")
    tvb_inference.simulation_setup(mpr, weights, sim_len, nsigma, BOLD_TR, dt, seed)
    print("Sbi inference")
    tvb_inference.sbi_infer()
    print("Run observed simulation")
    BOLD_obs = tvb_inference.run_sim(G)
    print("Posterior Distribution")
    tvb_inference.posterior_distribution(BOLD_obs, G, True)
