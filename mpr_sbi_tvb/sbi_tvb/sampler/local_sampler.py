import os
from subprocess import Popen, PIPE


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
