import time
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import sbi
import sbi.inference
import sbi.utils as utils
import sbi_tvb
import scipy
import scipy.stats
import torch
from matplotlib import rcParams
from sbi.inference import prepare_for_sbi, simulate_for_sbi
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from scipy import signal
from scipy.stats import kurtosis
from scipy.stats import moment
from scipy.stats import skew
from sbi_tvb import analysis, simulation
from tvb.simulator.lab import *

LOG = get_logger('demo')
rcParams['figure.figsize'] = 15, 6

#
# Define data an results dir
############################

sbi_tvb_path = os.path.dirname(os.path.dirname(sbi_tvb.__file__))
bold = np.loadtxt(os.path.join(sbi_tvb_path, 'data_input_files', 'meanTS_GS_bptf_Schaefer100_7NW.txt'))
weights = np.loadtxt(os.path.join(sbi_tvb_path, 'data_input_files', 'SC_Schaefer7NW100p_nolog10.txt'))
BOLD_TR = 2250

Res_dir = 'results'
if not os.path.exists(Res_dir):
    os.makedirs(Res_dir)


#
# Define the simulation function via TVB backend and time rescaling.
# The BOLD is derived moving average of the R signal
#
def run_sim(G, nsigma, weights, sim_len, BOLD_TR):
    t0 = time.time()

    #     jul                                       = data.Julich()
    #     subjs                                     = jul.list_subjects()
    #     subj_age,gender,education,subj_ID,_,_,_,_ = jul.metadata()
    magic_number = 124538.470647693

    #     SUBJ_TARG     = [subj_loc for subj_loc in range(len(subj_ID)) if mysubj in subj_ID[subj_loc] ][0]
    #     myage         = subj_age[SUBJ_TARG]

    #     _, weights    = jul.load_subject_sc_100(mysubj)
    NHALF = int(weights.shape[0] / 2)

    #     print(mysubj,flush=True)
    #     print(subj_ID[SUBJ_TARG],flush=True)
    #     print(myage,flush=True)
    #     print(str(tau),flush=True)
    #     print(str(G),flush=True)

    weights_orig = weights / magic_number
    weights_symm = weights_orig

    conn = connectivity.Connectivity(
        weights=weights_symm,
        region_labels=np.array(np.zeros(np.shape(weights_symm)[0]), dtype='<U128'),
        tract_lengths=np.zeros(np.shape(weights_symm)),
        areas=np.zeros(np.shape(weights_symm)[0]),
        speed=np.array(np.Inf, dtype=float),
        centres=np.zeros(np.shape(weights_symm)[0]))  # default 76 regions
    # conn_speed         = np.Inf
    # conn.weights       = weights_symm
    # conn.areas         = np.zeros(np.shape(weights_symm))
    # conn.tract_lengths = np.zeros(np.shape(weights_symm))
    #     print('weight:',weights_symm)
    #     print('shape :',np.shape(conn.weights))
    #     print('conn :', conn.weights)

    mpr = models.MontbrioPazoRoxin(
        eta=np.r_[-4.6],
        J=np.r_[14.5],
        Delta=np.r_[0.7],
        tau=np.r_[1],
    )
    #     mpr.state_variable_range['r'] = np.array([0.,.25])

    sim = simulator.Simulator(model=mpr,
                              connectivity=conn,
                              coupling=coupling.Scaling(
                                  a=np.r_[G]
                              ),
                              conduction_speed=np.Inf,
                              integrator=integrators.HeunStochastic(
                                  dt=dt,
                                  noise=noise.Additive(
                                      nsig=np.r_[nsigma, nsigma * 2],
                                      noise_seed=seed
                                  )
                              ),
                              monitors=[
                                  monitors.TemporalAverage(period=0.1),
                              ]
                              )

    sim.configure()

    (TemporalAverage_time, TemporalAverage_data), = simulation.run_nbMPR_backend(sim, simulation_length=sim_len)
    TemporalAverage_time *= 10  # rescale time

    #     Bold_time, Bold_data = simulation.tavg_to_bold(TemporalAverage_time, TemporalAverage_data, tavg_period=1., connectivity=sim.connectivity, svar=0, decimate=2000)

    R_TAVG = TemporalAverage_data[:, 0, :, 0]
    #     V = TemporalAverage_data[:,1,:,0]

    R = scipy.signal.decimate(R_TAVG, BOLD_TR, n=None, ftype='fir', axis=0)

    #     CPU_TIME = time.time() - t0
    #     print(['CPU time-->',CPU_TIME])

    return R.T


#
# Simulation setup
#
dt = 0.005
eta = -4.6
J = 14.5
Delta = 0.7
tau = 1.7
nsigma = 0.035
seed = 42
sim_len = 30e3
G = 2.45


#
# Compute the summary statistics via numpy and scipy.
# The function here extracts 10 momenta for each bold channel, FC mean, FCD mean, variance
# difference and standard deviation of FC stream.
# Check that you can compute FCD features via proper FCD packages
#
def calculate_summary_statistics(x, nn, bold_dt, features):
    """Calculate summary statistics

    Parameters
    ----------
    x : output of the simulator

    Returns
    -------
    np.array, summary statistics
    """

    #     X = np.reshape(x,(nn, int(x.shape[0]/nn)))
    X = x.reshape(nn, int(x.shape[0] / nn))

    n_summary = 16 * nn + (nn * nn) + 300 * 300

    wwidth = 30
    maxNwindows = 200
    olap = 0.96

    sum_stats_vec = np.concatenate((np.mean(X, axis=1),
                                    np.median(X, axis=1),
                                    np.std(X, axis=1),
                                    skew(X, axis=1),
                                    kurtosis(X, axis=1),
                                    ))

    #     sum_stats_vec = []

    for item in features:

        if item == 'higher_moments':
            sum_stats_vec = np.concatenate((sum_stats_vec,
                                            moment(X, moment=2, axis=1),
                                            moment(X, moment=3, axis=1),
                                            moment(X, moment=4, axis=1),
                                            moment(X, moment=5, axis=1),
                                            moment(X, moment=6, axis=1),
                                            moment(X, moment=7, axis=1),
                                            moment(X, moment=8, axis=1),
                                            moment(X, moment=9, axis=1),
                                            moment(X, moment=10, axis=1),
                                            ))

        if item == 'FC_corr':
            FC = np.corrcoef(X)
            off_diag_sum_FC = np.sum(FC) - np.trace(FC)
            # eigen_vals_FC, _ = LA.eig(FC)
            # pca = PCA(n_components=3)
            # PCA_FC = pca.fit_transform(FC)
            print('FC_Corr')
            sum_stats_vec = np.concatenate((sum_stats_vec,
                                            np.array([off_diag_sum_FC]),
                                            ))

        if item == 'FCD_corr':
            #                         FCDcorr,Pcorr,shift=extract_FCD(X,wwidth,maxNwindows,olap,mode='corr')

            win_FCD = 40e3
            NHALF = int(nn / 2)

            mask_inter = np.zeros([nn, nn])
            mask_inter[0:NHALF, NHALF:NHALF * 2] = 1
            mask_inter[NHALF:NHALF * 2, 0:NHALF] = 1

            bold_summ_stat = X.T

            FCD, fc_stack, speed_fcd = analysis.compute_fcd(bold_summ_stat, win_len=int(win_FCD / bold_dt), win_sp=1)
            fcd_inter, fc_stack_inter, _ = analysis.compute_fcd_filt(bold_summ_stat, mask_inter,
                                                                     win_len=int(win_FCD / bold_dt), win_sp=1)

            FCD_TRIU = np.triu(FCD, k=1)

            FCD_INTER_TRIU = np.triu(fcd_inter, k=1)

            FCD_MEAN = np.mean(FCD_TRIU)
            FCD_VAR = np.var(FCD_TRIU)
            FCD_OSC = np.std(fc_stack)
            FCD_OSC_INTER = np.std(fc_stack_inter)

            FCD_MEAN_INTER = np.mean(FCD_INTER_TRIU)
            FCD_VAR_INTER = np.var(FCD_INTER_TRIU)

            DIFF_VAR = FCD_VAR_INTER - FCD_VAR

            sum_stats_vec = np.concatenate((sum_stats_vec,
                                            np.array([FCD_MEAN]), np.array([FCD_OSC_INTER]), np.array([DIFF_VAR])
                                            ))

    sum_stats_vec = sum_stats_vec[0:n_summary]

    return sum_stats_vec


# Define the wrapper such that you can iterate the simulator with SBI
def MPR_simulator_wrapper(params):
    params = np.asarray(params)

    # params_alpha=params[0]
    params_G = params[0]

    BOLD_r_sim = run_sim(params_G, nsigma, weights, sim_len, BOLD_TR)

    summstats = torch.as_tensor(calculate_summary_statistics(BOLD_r_sim.reshape(-1), nn, BOLD_TR,
                                                             features=['higher_moments', 'FC_corr', 'FCD_corr']))

    return summstats


# Inference procedure. Although the inference function is defined in the SBI toolbox, the function below shows
# that you can potentially split the simulation step from the inference in case that it is needed.
# For example, the simulation time for the wrappper is too long and you might want to parallelize on HPC facilities.
# The inference function produces a posterorior object, which contains a neural network for posterior density estimation
def myinfer(
        simulator: Callable, prior, method: str, num_simulations: int, num_workers: int = 1
) -> NeuralPosterior:
    try:
        method_fun: Callable = getattr(sbi.inference, method.upper())
    except AttributeError:
        raise NameError(
            "Method not available. `method` must be one of 'SNPE', 'SNLE', 'SNRE'."
        )

    simulator, prior = prepare_for_sbi(simulator, prior)

    inference = method_fun(prior)
    theta, x = simulate_for_sbi(
        simulator=simulator,
        proposal=prior,
        num_simulations=num_simulations,
        num_workers=num_workers,
        show_progress_bar=True,
    )

    print(theta, flush=True)
    print(theta.shape, flush=True)
    print(x, flush=True)
    print(x.shape, flush=True)

    _ = inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior()

    mysavepath = os.path.join(Res_dir, 'inference_theta_jn_sim.npz')
    np.savez(mysavepath, theta=theta, x=x)

    return posterior


alpha_true = 0.0
beta_true = 0.0
G_true = 2.45
labels_params = [r'$G$']
params_true = np.hstack([alpha_true, G_true])

if __name__ == '__main__':
    print(
        "Check if the simulation with backend works and the CPU time for each run. "
        "According to Meysam's formalism, each signal matrix MUST BE (NODES X TIME).")
    start_time = time.time()
    BOLD_r = run_sim(G, nsigma, weights, sim_len, BOLD_TR)
    # bold_data = run_sim(G,nsigma,mysubj,alpha,sim_len)
    print(" single sim (sec) takes:", (time.time() - start_time))
    print(BOLD_r.shape)
    nt = BOLD_r.shape[1]
    nn = BOLD_r.shape[0]
    print(nn)
    print(nt)

    print("Compute the summary statistics via numpy and scipy. "
          "The function here extracts 10 momenta for each bold channel, "
          "FC mean, FCD mean, variance difference and standard deviation of FC stream. "
          "Check that you can compute FCD features via proper FCD packages")
    bold_summary_statistics = calculate_summary_statistics(BOLD_r.reshape(-1), nn, BOLD_TR,
                                                           features=['higher_moments', 'FC_corr', 'FCD_corr'])
    print(bold_summary_statistics.shape)

    print("Define the uniform prior and store it in a utils function for the SBI inference")
    prior_min_alpha = 0.
    prior_min_G = 1.5 * np.ones(1)
    prior_max_alpha = 1.
    prior_max_G = 3.2 * np.ones(1)
    prior_min = np.hstack([prior_min_G])
    prior_max = np.hstack([prior_max_G])
    print(prior_min.shape, prior_max.shape)
    prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max))
    print(prior)

    print("Run infer")
    start_time = time.time()
    posterior = myinfer(MPR_simulator_wrapper, prior, method='SNPE', num_simulations=20, num_workers=1)
    print("-" * 60)
    print("--- %s seconds ---" % (time.time() - start_time))

    print("SUMMARY STATISTICS TO GET THE POSTERIOR")
    alpha_true = 0.0
    beta_true = 0.0
    G_true = 2.45
    labels_params = [r'$G$']
    BOLD_obs = run_sim(G_true, nsigma, weights, sim_len, BOLD_TR)
    obs_summary_statistics = calculate_summary_statistics(BOLD_obs.reshape(-1), nn, 2250,
                                                          features=['higher_moments', 'FC_corr', 'FCD_corr'])
    print(BOLD_obs.shape, obs_summary_statistics.shape)

    print("Posterior distribution")
    params_true = np.hstack([G_true])
    num_samples = 1000
    posterior_samples = posterior.sample((num_samples,), obs_summary_statistics, sample_with_mcmc=True).numpy()
    mysavepath = os.path.join(Res_dir, 'posterior_samples_jn_sim.npz')
    np.savez(mysavepath, posterior_samples=posterior_samples)
    mysavepath = os.path.join(Res_dir, 'inference_theta_jn_sim.npz')
    myinference = np.load(mysavepath)
    theta = myinference['theta']
    x = myinference['x']
    print(x)

    print("Plot G posterior")
    mysavepath = os.path.join(Res_dir, 'posterior_samples_jn_sim.npz')
    myposterior = np.load(mysavepath)
    posterior_samples = myposterior['posterior_samples']
    G_posterior = posterior_samples[:, 0]
    plt.figure(figsize=(4, 4))
    parts = plt.violinplot(G_posterior, widths=0.7, showmeans=True, showextrema=True)
    plt.plot(1, params_true[0], 'o', color='k', alpha=0.9, markersize=8)
    plt.ylabel(' Posterior ' + r'${(G)}$', fontsize=18)
    plt.xlabel(r'${G}$', fontsize=18)
    plt.xticks([])
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()
