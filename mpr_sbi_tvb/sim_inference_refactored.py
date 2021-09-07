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

    print("Build TvbInference object")
    tvb_inference = TvbInference('results', num_simulations=10, num_workers=1)
    print("Build prior")
    tvb_inference.build_prior(1.5, 3.2)
    print("Simulation setup")
    tvb_inference.simulation_setup_default(weights, sim_len, nsigma, BOLD_TR, dt, seed)
    print("Sbi inference")
    tvb_inference.sbi_infer()
    print("Run observed simulation")
    BOLD_obs = tvb_inference.run_sim(G)
    print("Posterior Distribution")
    found_value = tvb_inference.posterior_distribution(BOLD_obs, G, True)
    print("G value found: {}".format(found_value))

    loaded_simulations = np.load("/Users/bvalean/WORK/tvb-inversion/mpr_sbi_tvb/results/inference_theta_jn_sim.npz")
    theta_sim = loaded_simulations['theta']
    x_sim = loaded_simulations['x']
    tvb_inference.train_network(theta_sim, x_sim)
    print("Run observed simulation")
    BOLD_obs = tvb_inference.run_sim(G)
    print("Posterior Distribution")
    found_value = tvb_inference.posterior_distribution(BOLD_obs, G, True)
    print("G value found: {}".format(found_value))