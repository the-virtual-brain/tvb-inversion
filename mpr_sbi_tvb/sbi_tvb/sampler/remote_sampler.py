import os
from time import sleep

import pyunicore.client as unicore_client

import sbi_tvb
from sbi_tvb.sampler.local_samplers import DockerLocalSampler


class UnicoreConfig:

    def __init__(self, project, site=None, job_name=None, runtime_hours=1):
        if project is None or len(project.strip()) == 0:
            raise Exception("Please specify the HPC project to use for this job!")

        if site is None or len(site.strip()) == 0:
            raise Exception("Please specify the HPC site where the job should run!")

        self.project = project
        self.site = site
        self.job_name = job_name
        self.runtime_hours = runtime_hours
        self.number_of_nodes = 1
        self.node_constraints = 'mc'


class UnicoreSampler(DockerLocalSampler):
    HPC_SCRIPT = 'launch_simulation_hpc.sh'

    def __init__(self, num_simulations, num_workers, unicore_config):
        super(UnicoreSampler, self).__init__(num_simulations, num_workers)
        self.unicore_config = unicore_config

    def _retrieve_token(self):
        try:
            from clb_nb_utils import oauth as clb_oauth
            token = clb_oauth.get_token()
        except (ModuleNotFoundError, ConnectionError) as e:
            self.logger.warn(f"Could not connect to EBRAINS to retrieve an auth token: {e}")
            self.logger.info("Will try to use the auth token defined by environment variable CLB_AUTH...")

            token = os.environ.get('CLB_AUTH')
            if token is None:
                self.logger.error("No auth token defined as environment variable CLB_AUTH! Please define one!")
                raise Exception("Cannot connect to EBRAINS HPC without an auth token! Either run this on "
                                "Collab, or define the CLB_AUTH environment variable!")

            self.logger.info("Successfully retrieved the auth token from environment variable CLB_AUTH!")
        return token

    def _gather_inputs(self, dir_name):
        self.logger.info('Gathering files for stage in...')
        hpc_input_names = os.listdir(dir_name)
        hpc_input_paths = list()
        for input_name in hpc_input_names:
            hpc_input_paths.append(os.path.join(dir_name, input_name))

        sbi_tvb_path = os.path.dirname(sbi_tvb.__file__)
        script_path = os.path.join(sbi_tvb_path, self.HPC_SCRIPT)
        hpc_input_paths.append(script_path)

        return hpc_input_paths

    def _prepare_unicore_job(self, tvb_simulator):
        job_name = self.unicore_config.job_name
        if job_name is None or len(job_name.strip()) == 0:
            job_name = 'TVB-INVERSION_{}_{}'.format(self.num_simulations, self.num_workers)

        my_job = {
            'Executable': self.HPC_SCRIPT,
            'Arguments': [self.DOCKER_DATA_DIR, tvb_simulator.gid.hex, self.num_simulations, self.num_workers],
            'Project': self.unicore_config.project,
            'Name': job_name,
            'Resources': {'Nodes': self.unicore_config.number_of_nodes,
                          'NodeConstraints': self.unicore_config.node_constraints,
                          'Runtime': f'{self.unicore_config.runtime_hours}h'},
        }

        return my_job

    def _connect_unicore(self):
        self.logger.info(f'Connecting to {self.unicore_config.site} via PyUnicore...')
        # TODO: get token and site_url generically?
        token = self._retrieve_token()
        transport = unicore_client.Transport(token)
        all_sites = unicore_client.get_sites(transport)
        client = unicore_client.Client(transport, all_sites[self.unicore_config.site])

        return client

    def _monitor_job(self, job):
        self.logger.info('Starting to monitor job...')
        while job.is_running():
            self.logger.info(f'Job has status: {job.properties["status"]}')
            sleep(600)

    def _stage_out_results(self, job, result_path):
        self.logger.info('Trying to stage out the results...')
        wd = job.working_dir.listdir()
        try:
            wd[os.path.basename(result_path)].download(result_path)
            self.logger.info(f'Downloaded priors sampling results as {result_path}')
        except KeyError:
            raise Exception("The priors sampling results could not be downloaded from HPC! "
                            "Please check the logs on HPC to understand what went wrong!")

    def run(self, simulator, dir_name, result_name):
        hpc_inputs = self._gather_inputs(dir_name)
        job_config = self._prepare_unicore_job(simulator)

        client = self._connect_unicore()
        job = client.new_job(job_description=job_config, inputs=hpc_inputs)
        self.logger.info(f'Mount point on HPC for job is: {job.working_dir.properties["mountPoint"]}')

        self._monitor_job(job)

        result_path = os.path.join(dir_name, result_name)
        self._stage_out_results(job, result_path)

        theta, x = self.read_results(result_path)

        return theta, x
