import tempfile
from copy import deepcopy
from typing import Callable, List

import numpy as np
import sbi
import sbi.utils as utils
import scipy
import torch
from numpy import load
from sbi.inference import prepare_for_sbi, simulate_for_sbi
from sbi_tvb.prior import Prior
from sbi_tvb.utils import custom_setattr
from scipy import signal

from tvb.config.init.datatypes_registry import populate_datatypes_registry
from tvb.core.neocom.h5 import store_ht
from tvb.simulator.backend.nb_mpr import NbMPRBackend
from tvb.simulator.lab import *

from mpr_sbi_tvb.sbi_tvb.features import _calculate_summary_statistics
from mpr_sbi_tvb.sbi_tvb.sampler.remote_sampler import UnicoreSampler


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
            summary_statistics = _calculate_summary_statistics
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
        if self.backend is None:
            self.backend = NbMPRBackend

        if not self.preparing_for_sbi:
            print("Using params: {}".format(params))
            self._set_sim_params(used_simulator, params)
            temporal_average_time, temporal_average_data = self.submit_simulation(self.backend, used_simulator)
        else:
            temporal_average_time, temporal_average_data = self._submit_simulation_local(self.backend, used_simulator)

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

    def _submit_simulation_remote(self, backend, tvb_simulator):
        """
        Run TVB simulation remote on a HPC cluster.
        """
        import tempfile

        dir_name = tempfile.mkdtemp(prefix='simulator-', dir=os.getcwd())
        print(f'Using dir {dir_name} for gid {tvb_simulator.gid}')
        populate_datatypes_registry()

        store_ht(tvb_simulator, dir_name)

        ts_path = self._pyunicore_run(dir_name, tvb_simulator)

        with load(ts_path) as data:
            ts_data = data['data']
            ts_time = data['time']

        print(f"TODO: Submit {backend} and {tvb_simulator} to HPC backend")
        # StorageInterface.remove_folder(dir_name)
        return ts_time, ts_data

    def sample_priors_remote(self, num_simulations, num_workers, project):
        used_simulator = deepcopy(self.simulator)

        dir_name = tempfile.mkdtemp(prefix='simulator-', dir=os.getcwd())
        print(f'Using dir {dir_name} for gid {used_simulator.gid}')
        populate_datatypes_registry()

        store_ht(used_simulator, dir_name)

        remote_sampler = UnicoreSampler(num_simulations, num_workers, project)
        result = remote_sampler.run(dir_name, used_simulator, self.SIMULATIONS_RESULTS)

        with load(result) as f:
            theta = f['theta']
            x = f['x']

        self.theta = theta
        self.x = x

        return theta, x

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
        # self.theta = theta
        # self.x = x
        if save_path is None:
            save_path = os.getcwd()
        mysavepath = os.path.join(save_path, TvbInference.SIMULATIONS_RESULTS)
        print(f'Saving results at {mysavepath}...')
        np.savez(mysavepath, theta=theta, x=x)
        print(f'Results saved!')
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
