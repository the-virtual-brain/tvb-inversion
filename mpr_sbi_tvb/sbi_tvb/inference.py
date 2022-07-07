import os
from copy import deepcopy
from subprocess import Popen, PIPE
from time import sleep
from typing import Callable, List

import numpy as np
import sbi
import sbi.utils as utils
import scipy
import torch
from numpy import load
from sbi.inference import prepare_for_sbi, simulate_for_sbi
from sbi_tvb import analysis
from sbi_tvb.prior import Prior
from sbi_tvb.utils import custom_setattr
from scipy import signal
from scipy.stats import kurtosis
from scipy.stats import moment
from scipy.stats import skew
from tvb.config.init.datatypes_registry import populate_datatypes_registry
from tvb.core.neocom.h5 import store_ht, load_ht
from tvb.simulator.backend.nb_mpr import NbMPRBackend
from tvb.simulator.lab import *
from tvb.storage.storage_interface import StorageInterface


class TvbInference:
    SIMULATIONS_RESULTS = "inference_theta_jn_sim.npz"
    POSTERIOR_SAMPLES = "posterior_samples_jn_sim.npy"

    def __init__(self, sim: simulator.Simulator,
                 priors: List[Prior],
                 summary_statistics: Callable = None,
                 remote: bool = False):
        """
        Parameters
        -----------------------
        sim: simulator.Simulator
            TVB simulator to be used in inference

        priors: List[Prior]
            list of priors. Define min, max of inferred attributes

        summary_statistics: Callable
            custom function used to reduce dimension. This function which takes as input TVB simulator output and
            returns an array
        """
        populate_datatypes_registry()
        self.simulator = sim
        self.prior = self._build_prior(priors)
        self.priors_list = priors
        if summary_statistics is None:
            summary_statistics = self._calculate_summary_statistics
        self.summary_statistics = summary_statistics
        self.backend = None
        self.theta = None
        self.x = None
        self.trained = False
        self.preparing_for_sbi = False
        self.inf_posterior = None
        self.submit_simulation = self._submit_simulation_local
        if remote:
            self.submit_simulation = self._submit_simulation_remote

    def _build_prior(self, priors: List[Prior]):
        """
        Build pytorch prior based on priors list

        Parameters
        ----------
        priors: List[Prior]
        """
        prior_min = None
        prior_max = None
        for prior in priors:
            min_value = prior.min
            max_value = prior.max
            if prior_min is None:
                prior_min = min_value
            else:
                prior_min = np.append(prior_min, min_value)

            if prior_max is None:
                prior_max = max_value
            else:
                prior_max = np.append(prior_max, max_value)

        return utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max))

    def _calculate_summary_statistics(self, x, features=None):
        """
        Calculate summary statistics via numpy and scipy.
        The function here extracts 10 momenta for each bold channel, FC mean, FCD mean, variance
        difference and standard deviation of FC stream.
        Check that you can compute FCD features via proper FCD packages

        Parameters
        ----------
        x : output of the simulator

        Returns
        -------
        np.array, summary statistics
        """
        if features is None:
            features = ['higher_moments', 'FC_corr', 'FCD_corr']
        nn = self.simulator.connectivity.weights.shape[0]

        X = x.reshape(nn, int(x.shape[0] / nn))

        n_summary = 16 * nn + (nn * nn) + 300 * 300
        bold_dt = 2250

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

    def _set_sim_params(self, sim: simulator.Simulator, params):
        index = 0
        for prior in self.priors_list:
            value = params[index:index + prior.size]
            index += prior.size
            custom_setattr(sim, prior.path, value)

    def run_sim(self, params):
        """
        Define the simulation function via TVB backend and time rescaling.
        The BOLD is derived moving average of the R signal

        Parameters
        ----------
        params: list of inferred parameters. These params will be set on the TVB simulator
        """
        used_simulator = deepcopy(self.simulator)
        if not self.preparing_for_sbi:
            print("Using params: {}".format(params))
            self._set_sim_params(used_simulator, params)
        if self.backend is None:
            self.backend = NbMPRBackend

        temporal_average_time, temporal_average_data = self.submit_simulation(self.backend, used_simulator)

        # TODO: Are these adjustments generic?
        temporal_average_time *= 10  # rescale time

        R_TAVG = temporal_average_data[:, 0, :, 0]

        R = scipy.signal.decimate(R_TAVG, 2250, n=None, ftype='fir', axis=0)

        return R.T

    def _submit_simulation_local(self, backend, tvb_simulator):
        """
        Run TVB simulation locally on the same machine.
        Parameters
        ----------
        backend: Backend used to run simulation. By default NbMPRBackend is used.
        tvb_simulator: TVB simulator with inferred parameters already set
        """
        tvb_simulator.configure()
        (temporal_average_time, temporal_average_data), = backend().run_sim(tvb_simulator,
                                                                            simulation_length=tvb_simulator.simulation_length)
        return temporal_average_time, temporal_average_data

    def _prepare_unicore_job(self, tvb_simulator):
        # "/home/data"
        docker_dir_name = '/home/data'

        SH_SCRIPT = 'launch_simulation_hpc.sh'
        script_path = os.path.join(os.getcwd(), 'sbi_tvb', SH_SCRIPT)

        HPC_PROJECT = 'ich012'

        my_job = {
            'Executable': os.path.basename(script_path),
            'Arguments': [docker_dir_name, tvb_simulator.gid.hex],
            'Project': HPC_PROJECT
        }

        return my_job, script_path

    def _run_local_docker_job(self, dir_name, tvb_simulator):
        docker_dir_name = '/home/data'
        script_path = os.path.join(os.getcwd(), 'sbi_tvb', 'launch_simulation_docker.sh')
        run_params = ['bash', script_path, dir_name, docker_dir_name, tvb_simulator.gid.hex]

        launched_process = Popen(run_params, stdout=PIPE, stderr=PIPE)

        subprocess_result = launched_process.communicate()
        returned = launched_process.wait()

        if returned != 0:
            print(f"Failed to launch job")
            return

    def __retrieve_token(self):
        try:
            from clb_nb_utils import oauth as clb_oauth
            token = clb_oauth.get_token()
        except (ModuleNotFoundError, ConnectionError) as e:
            print(f"Could not connect to EBRAINS to retrieve an auth token: {e}")
            print("Will try to use the auth token defined by environment variable CLB_AUTH...")

            token = os.environ.get('CLB_AUTH')
            if token is None:
                print("No auth token defined as environment variable CLB_AUTH! Please define one!")
                raise Exception("Cannot connect to EBRAINS HPC without an auth token! Either run this on "
                                             "Collab, or define the CLB_AUTH environment variable!")

            print("Successfully retrieved the auth token from environment variable CLB_AUTH!")
        return token

    def _submit_simulation_remote(self, backend, tvb_simulator):
        """
        Run TVB simulation remote on a HPC cluster.
        """
        import tempfile
        import pyunicore.client as unicore_client

        dir_name = tempfile.mkdtemp(prefix='simulator-', dir=os.getcwd())
        print(f'Using dir {dir_name} for gid {tvb_simulator.gid}')
        store_ht(tvb_simulator, dir_name)

        hpc_input_names = os.listdir(dir_name)
        hpc_input_paths = list()
        for input_name in hpc_input_names:
            hpc_input_paths.append(os.path.join(dir_name, input_name))

        job_config, script = self._prepare_unicore_job(tvb_simulator)
        hpc_input_paths.append(script)

        # TODO: get token and site_url generically?
        token = self.__retrieve_token()
        transport = unicore_client.Transport(token)
        all_sites = unicore_client.get_sites(transport)
        client = unicore_client.Client(transport, all_sites['DAINT-CSCS'])

        job = client.new_job(job_description=job_config, inputs=hpc_input_paths)

        # # TODO: monitor and wait for job
        while job.is_running():
            sleep(60)

        # # TODO: stage out results, read them and return arrays
        ts_name = 'time_series.npz'
        ts_path = os.path.join(dir_name, ts_name)
        wd = job.working_dir.listdir()
        wd[ts_name].download(ts_path)

        with load(ts_path) as data:
            ts_data = data['data']
            ts_time = data['time']

        print(f"TODO: Submit {backend} and {tvb_simulator} to HPC backend")
        StorageInterface.remove_folder(dir_name)
        return ts_time, ts_data

    def _MPR_simulator_wrapper(self, params):
        """
        Define the wrapper such that you can iterate the simulator with SBI
        """
        params = np.asarray(params)
        BOLD_r_sim = self.run_sim(params)
        return torch.as_tensor(self.summary_statistics(BOLD_r_sim.reshape(-1)))

    def _simulate_for_sbi(self, sim, prior, num_simulations, num_workers, save_path=None):
        theta, x = simulate_for_sbi(
            simulator=sim,
            proposal=prior,
            num_simulations=num_simulations,
            num_workers=num_workers,
            show_progress_bar=True,
        )
        print(f'Theta shape is {theta.shape}, x shape is {x.shape}')
        self.theta = theta
        self.x = x
        if save_path is None:
            save_path = os.getcwd()
        mysavepath = os.path.join(save_path, TvbInference.SIMULATIONS_RESULTS)
        np.savez(mysavepath, theta=theta, x=x)
        return theta, x

    def sample_priors(self, backend=NbMPRBackend, save_path=None, num_simulations=20, num_workers=1):
        """
        Inference procedure. Although the inference function is defined in the SBI toolbox, the function below shows
        that you can potentially split the simulation step from the inference in case that it is needed.
        For example, the simulation time for the wrappper is too long and you might want to parallelize on HPC facilities.
        The inference function produces a posterorior object, which contains a neural network for posterior density estimation
        """
        self.backend = backend
        self.preparing_for_sbi = True
        sim, prior = prepare_for_sbi(self._MPR_simulator_wrapper, self.prior)
        self.preparing_for_sbi = False
        theta, x = self._simulate_for_sbi(sim, prior, num_simulations, num_workers, save_path)
        return theta, x

    def train(self, method='SNPE', load_path=None):
        """
        Train neural network

        Parameters
        ----------
        method: str
            SBI method. Must be one of 'SNPE', 'SNLE', 'SNRE'
        load_path: str
            Path to the file which contains info about theta and simulator output.
        """
        try:
            method_fun: Callable = getattr(sbi.inference, method.upper())
        except AttributeError:
            raise NameError(
                "Method not available. `method` must be one of 'SNPE', 'SNLE', 'SNRE'."
            )
        if self.prior is None:
            raise Exception("Prior is not defined")

        theta = self.theta
        x = self.x
        if load_path is not None:
            loaded_simulations = np.load(load_path)
            theta = loaded_simulations['theta']
            x = loaded_simulations['x']

        if not isinstance(theta, torch.Tensor):
            theta = torch.as_tensor(theta)

        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)

        self.preparing_for_sbi = True
        sim, prior = prepare_for_sbi(self._MPR_simulator_wrapper, self.prior)
        self.preparing_for_sbi = False
        inference = method_fun(prior)
        _ = inference.append_simulations(theta, x).train()
        inf_posterior = inference.build_posterior()

        self.inf_posterior = inf_posterior
        self.trained = True
        return inf_posterior

    def posterior(self, data, save_path=None):
        """
        Run actual inference

        Parameters
        ----------
        data: numpy array
            TS which will be inferred
        save_path: str
            directory where to save the results. If it is not set, current directory will be used.

        """
        if not self.trained:
            raise Exception("You have to train the neural network before generating distribution")

        obs_summary_statistics = self.summary_statistics(data.reshape(-1))
        num_samples = 1000
        posterior_samples = self.inf_posterior.sample((num_samples,), obs_summary_statistics,
                                                      sample_with='mcmc').numpy()
        if save_path is None:
            save_path = os.getcwd()
        mysavepath = os.path.join(save_path, TvbInference.POSTERIOR_SAMPLES)
        np.save(mysavepath, posterior_samples)
        return posterior_samples
