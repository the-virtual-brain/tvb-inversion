import os
from subprocess import Popen, PIPE

import numpy as np
from sbi.inference import simulate_for_sbi

from sbi_tvb.logger.builder import get_logger


class BaseSampler(object):

    def __init__(self, num_simulations, num_workers):
        self.logger = get_logger(self.__class__.__module__)
        self.num_simulations = num_simulations
        self.num_workers = num_workers

    @staticmethod
    def read_results(result):
        with np.load(result) as f:
            theta = f['theta']
            x = f['x']

        return theta, x


class LocalSampler(BaseSampler):

    def run(self, simulator, prior, dir_name, result_name):
        theta, x = simulate_for_sbi(
            simulator=simulator,
            proposal=prior,
            num_simulations=self.num_simulations,
            num_workers=self.num_workers,
            show_progress_bar=True,
        )
        self.logger.info(f'Theta shape is {theta.shape}, x shape is {x.shape}')

        if dir_name is None:
            dir_name = os.getcwd()
        mysavepath = os.path.join(dir_name, result_name)
        self.logger.info(f'Saving results at {mysavepath}...')

        np.savez(mysavepath, theta=theta, x=x)
        self.logger.info(f'Results saved!')

        return theta, x


class DockerLocalSampler(BaseSampler):
    SH_SCRIPT = 'launch_simulation_docker.sh'
    DOCKER_DATA_DIR = '/home/data'

    def run(self, simulator, dir_name, result_name):
        script_path = os.path.join(os.getcwd(), 'sbi_tvb', self.SH_SCRIPT)
        run_params = ['bash', script_path, dir_name, self.DOCKER_DATA_DIR, simulator.gid.hex, self.num_simulations,
                      self.num_workers]

        launched_process = Popen(run_params, stdout=PIPE, stderr=PIPE)

        launched_process.communicate()
        returned = launched_process.wait()

        if returned != 0:
            self.logger.error(f"Failed to launch job")
            return

        theta, x = self.read_results(result_name)

        return theta, x
