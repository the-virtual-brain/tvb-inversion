import os
from subprocess import Popen, PIPE
from time import sleep


class LocalSampler(object):

    def __init__(self):
        pass


class DockerLocalSampler(LocalSampler):

    def __init__(self):
        pass

    def _local_docker_run(self, dir_name, tvb_simulator):
        run_params = ['sh',
                      '/Users/pipeline/WORK/TVB_GIT/tvb-inversion/tvb-inversion/mpr_sbi_tvb/sbi_tvb/launch_simulation_docker.sh',
                      dir_name, tvb_simulator.gid.hex]

        launched_process = Popen(run_params, stdout=PIPE, stderr=PIPE)

        subprocess_result = launched_process.communicate()
        print(f"Finished with launch of operation")
        returned = launched_process.wait()

        if returned != 0:
            print(f"Operation suffered fatal failure! {subprocess_result}")

        del launched_process

        ts_name = 'time_series.npz'
        ts_path = os.path.join(dir_name, ts_name)

        return ts_path

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


class UnicoreSampler(object):
    HPC_SCRIPT = 'launch_simulation_hpc.sh'

    def __init__(self, num_simulations, num_workers, project):
        self.num_simulations = num_simulations
        self.num_workers = num_workers
        self.project = project

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

    def _gather_inputs(self, dir_name):
        hpc_input_names = os.listdir(dir_name)
        hpc_input_paths = list()
        for input_name in hpc_input_names:
            hpc_input_paths.append(os.path.join(dir_name, input_name))

        script_path = os.path.join(os.getcwd(), 'sbi_tvb', self.HPC_SCRIPT)
        hpc_input_paths.append(script_path)

        return hpc_input_paths

    def _prepare_unicore_job(self, tvb_simulator):
        docker_dir_name = '/home/data'

        # TODO: specify resources in pyunicore instead of srun
        my_job = {
            'Executable': self.HPC_SCRIPT,
            'Arguments': [docker_dir_name, tvb_simulator.gid.hex, self.num_simulations, self.num_workers],
            'Project': self.project,
            'Name': 'TVB-INVERSION_{}_{}'.format(self.num_simulations, self.num_workers),
            # 'Resources': {'Runtime': '23h'},
        }

        return my_job

    def _connect_unicore(self):
        import pyunicore.client as unicore_client

        # TODO: get token and site_url generically?
        token = self.__retrieve_token()
        transport = unicore_client.Transport(token)
        all_sites = unicore_client.get_sites(transport)
        client = unicore_client.Client(transport, all_sites['DAINT-CSCS'])

        return client

    def _monitor_job(self, job):
        while job.is_running():
            print(job.properties['status'])
            sleep(60)

    def _stage_out_results(self, job, result_path):
        wd = job.working_dir.listdir()
        try:
            wd[os.path.basename(result_path)].download(result_path)
            print(f'Downloaded sampling result as {result_path}')
        except KeyError:
            raise Exception("The priors sampling results could not be downloaded from HPC! "
                            "Please check the logs on HPC to understand what went wrong!")

    def run(self, dir_name, tvb_simulator, result_name):
        hpc_inputs = self._gather_inputs(dir_name)
        job_config = self._prepare_unicore_job(tvb_simulator)

        client = self._connect_unicore()
        job = client.new_job(job_description=job_config, inputs=hpc_inputs)
        print(job.working_dir.properties['mountPoint'])

        self._monitor_job(job)

        result_path = os.path.join(dir_name, result_name)
        self._stage_out_results(job, result_path)

        return result_path
