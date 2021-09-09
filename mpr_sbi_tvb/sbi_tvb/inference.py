from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import sbi
import sbi.utils as utils
import scipy
import torch
from sbi.inference import prepare_for_sbi, simulate_for_sbi
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi_tvb import analysis, simulation
from scipy import signal
from scipy.stats import kurtosis
from scipy.stats import moment
from scipy.stats import skew
from tvb.simulator.lab import *
from tvb.simulator.models.base import Model


class TvbInference:
    SIMULATIONS_RESULTS = "inference_theta_jn_sim.npz"
    POSTERIOR_SAMPLES = "posterior_samples_jn_sim.npz"

    def __init__(self, results_dir, method='SNPE', num_simulations=20, num_workers=1):
        self.method = method
        self.num_simulations = num_simulations
        self.num_workers = num_workers
        self.results_dir = results_dir
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        self.posterior = None
        self.prior = None

        # Simulations config
        self.model = None
        self.conn = None
        self.cond_speed = None
        self.integrator = None
        self.monitors = None
        self.weights = None
        self.sim_len = None
        self.nsigma = None
        self.BOLD_TR = None
        self.dt = None
        self.seed = None

        self.trained = False

    @property
    def results_directory(self):
        if os.path.isabs(self.results_dir):
            return self.results_dir
        return os.path.join(os.getcwd(), self.results_dir)

    @property
    def sim_results_path(self):
        return os.path.join(self.results_directory, TvbInference.SIMULATIONS_RESULTS)

    @property
    def posterior_samples_path(self):
        return os.path.join(self.results_directory, TvbInference.POSTERIOR_SAMPLES)

    def _validate_configs(self):
        if (self.weights is None and self.conn is None) or \
                self.sim_len is None or \
                (self.nsigma is None and self.integrator is None) or \
                (self.seed is None and self.integrator is None) or \
                (self.dt is None and self.integrator is None) or \
                self.BOLD_TR is None or \
                self.prior is None:
            return False
        return True

    def simulation_setup_default(self, weights, sim_len, nsigma, BOLD_TR, dt, seed):
        self.trained = False
        self.weights = weights
        self.sim_len = sim_len
        self.nsigma = nsigma
        self.BOLD_TR = BOLD_TR
        self.dt = dt
        self.seed = seed

    def simulation_setup_tvb(self, model: models.base.Model = None, connectivity: connectivity.Connectivity = None,
                             cond_speed: float = None, integrator: integrators.Integrator = None, monitors: [] = None,
                             sim_len: int = None, bold_tr: int = None):
        self.trained = False
        self.model = model
        self.conn = connectivity
        self.cond_speed = cond_speed
        self.integrator = integrator
        self.monitors = monitors
        self.sim_len = sim_len
        self.BOLD_TR = bold_tr

    def build_prior(self, prior_min, prior_max):
        self.trained = False
        # Min value
        prior_min = prior_min * np.ones(1)
        prior_min = np.hstack([prior_min])

        # Max value
        prior_max = prior_max * np.ones(1)
        prior_max = np.hstack([prior_max])

        self.prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max))

    #
    # Compute the summary statistics via numpy and scipy.
    # The function here extracts 10 momenta for each bold channel, FC mean, FCD mean, variance
    # difference and standard deviation of FC stream.
    # Check that you can compute FCD features via proper FCD packages
    #
    def _calculate_summary_statistics(self, x, nn, bold_dt, features):
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

        sum_stats_vec = np.concatenate((np.mean(X, axis=1),
                                        np.median(X, axis=1),
                                        np.std(X, axis=1),
                                        skew(X, axis=1),
                                        kurtosis(X, axis=1),
                                        ))

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
                print('FC_Corr')
                sum_stats_vec = np.concatenate((sum_stats_vec,
                                                np.array([off_diag_sum_FC]),
                                                ))

            if item == 'FCD_corr':
                win_FCD = 40e3
                NHALF = int(nn / 2)

                mask_inter = np.zeros([nn, nn])
                mask_inter[0:NHALF, NHALF:NHALF * 2] = 1
                mask_inter[NHALF:NHALF * 2, 0:NHALF] = 1

                bold_summ_stat = X.T

                FCD, fc_stack, speed_fcd = analysis.compute_fcd(bold_summ_stat, win_len=int(win_FCD / bold_dt),
                                                                win_sp=1)
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

    #
    # Define the simulation function via TVB backend and time rescaling.
    # The BOLD is derived moving average of the R signal
    #
    def run_sim(self, G):
        model = self.model
        if model is None:
            model = models.MontbrioPazoRoxin(
                eta=np.r_[-4.6],
                J=np.r_[14.5],
                Delta=np.r_[0.7],
                tau=np.r_[1],
            )

        conn = self.conn
        if conn is None:
            magic_number = 124538.470647693
            weights_orig = self.weights / magic_number
            conn = connectivity.Connectivity(
                weights=weights_orig,
                region_labels=np.array(np.zeros(np.shape(weights_orig)[0]), dtype='<U128'),
                tract_lengths=np.zeros(np.shape(weights_orig)),
                areas=np.zeros(np.shape(weights_orig)[0]),
                speed=np.array(np.Inf, dtype=float),
                centres=np.zeros(np.shape(weights_orig)[0]))  # default 76 regions

        integrator = self.integrator
        if integrator is None:
            integrator = integrators.HeunStochastic(
                dt=self.dt,
                noise=noise.Additive(
                    nsig=np.r_[self.nsigma, self.nsigma * 2],
                    noise_seed=self.seed
                )
            )

        _monitors = self.monitors
        if _monitors is None:
            _monitors = [monitors.TemporalAverage(period=0.1)]

        cond_speed = self.cond_speed
        if cond_speed is None:
            cond_speed = np.Inf

        sim = simulator.Simulator(model=model,
                                  connectivity=conn,
                                  coupling=coupling.Scaling(
                                      a=np.r_[G]
                                  ),
                                  conduction_speed=cond_speed,
                                  integrator=integrator,
                                  monitors=_monitors
                                  )

        sim.configure()

        (TemporalAverage_time, TemporalAverage_data), = simulation.run_nbMPR_backend(sim,
                                                                                     simulation_length=self.sim_len)
        TemporalAverage_time *= 10  # rescale time

        R_TAVG = TemporalAverage_data[:, 0, :, 0]

        R = scipy.signal.decimate(R_TAVG, self.BOLD_TR, n=None, ftype='fir', axis=0)

        return R.T

    # Define the wrapper such that you can iterate the simulator with SBI
    def MPR_simulator_wrapper(self, params):
        params = np.asarray(params)

        params_G = params[0]
        BOLD_r_sim = self.run_sim(params_G)
        nn = BOLD_r_sim.shape[0]
        return torch.as_tensor(self._calculate_summary_statistics(BOLD_r_sim.reshape(-1), nn, self.BOLD_TR,
                                                                  features=['higher_moments', 'FC_corr',
                                                                            'FCD_corr']))

    # Inference procedure. Although the inference function is defined in the SBI toolbox, the function below shows
    # that you can potentially split the simulation step from the inference in case that it is needed.
    # For example, the simulation time for the wrappper is too long and you might want to parallelize on HPC facilities.
    # The inference function produces a posterorior object, which contains a neural network for posterior density estimation
    def run_simulations(self):
        if not self._validate_configs():
            raise Exception("Please check simulation configs")

        sim, prior = prepare_for_sbi(self.MPR_simulator_wrapper, self.prior)

        theta, x = simulate_for_sbi(
            simulator=sim,
            proposal=prior,
            num_simulations=self.num_simulations,
            num_workers=self.num_workers,
            show_progress_bar=True,
        )
        mysavepath = os.path.join(self.results_dir, TvbInference.SIMULATIONS_RESULTS)
        np.savez(mysavepath, theta=theta, x=x)
        return theta, x

    def train_network(self, theta, x):
        try:
            method_fun: Callable = getattr(sbi.inference, self.method.upper())
        except AttributeError:
            raise NameError(
                "Method not available. `method` must be one of 'SNPE', 'SNLE', 'SNRE'."
            )
        if self.prior is None:
            raise Exception("Prior is not defined")

        if not isinstance(theta, torch.Tensor):
            theta = torch.as_tensor(theta)

        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)

        sim, prior = prepare_for_sbi(self.MPR_simulator_wrapper, self.prior)
        inference = method_fun(prior)
        _ = inference.append_simulations(theta, x).train()
        posterior = inference.build_posterior()

        self.posterior = posterior
        self.trained = True
        return posterior

    def sbi_infer(self) -> NeuralPosterior:

        theta, x = self.run_simulations()
        return self.train_network(theta, x)

    def posterior_distribution(self, observed_bold, G_true=None, plot_posterior=False):
        if not self.trained:
            raise Exception("You have to train the neural network before generating distribution")

        nn = observed_bold.shape[0]
        obs_summary_statistics = self._calculate_summary_statistics(observed_bold.reshape(-1), nn, 2250,
                                                                    features=['higher_moments', 'FC_corr', 'FCD_corr'])
        num_samples = 1000
        posterior_samples = self.posterior.sample((num_samples,), obs_summary_statistics, sample_with_mcmc=True).numpy()
        mysavepath = os.path.join(self.results_dir, TvbInference.POSTERIOR_SAMPLES)
        np.savez(mysavepath, posterior_samples=posterior_samples)
        G_posterior = posterior_samples[:, 0]
        if plot_posterior:
            print("Plot G posterior")
            params_true = np.hstack([G_true])
            plt.figure(figsize=(4, 4))
            plt.violinplot(G_posterior, widths=0.7, showmeans=True, showextrema=True)
            plt.plot(1, params_true[0], 'o', color='k', alpha=0.9, markersize=8)
            plt.ylabel(' Posterior ' + r'${(G)}$', fontsize=18)
            plt.xlabel(r'${G}$', fontsize=18)
            plt.xticks([])
            plt.yticks(fontsize=14)
            plt.tight_layout()
            plt.show()

        return G_posterior.mean()
