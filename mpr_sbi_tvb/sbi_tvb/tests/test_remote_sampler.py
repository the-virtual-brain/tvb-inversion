import os
import pytest as pytest
from tvb.simulator.simulator import Simulator
from sbi_tvb.sampler.remote_sampler import UnicoreSampler, UnicoreConfig


class MockFilePath:
    def __init__(self, is_file=True):
        self.is_file = is_file

    def isfile(self):
        return self.is_file

    def download(self, file):
        pass


class WorkingDirMock:
    def __init__(self, dirs={}):
        self.dirs = dirs or {
            'file1': MockFilePath(),
        }

    def listdir(self):
        return self.dirs


class MockPyUnicoreJob:
    def __init__(self, job_url='test', isrunning=False, job_id='test'):
        self.job_url = job_url
        self.job_id = job_id
        self.working_dir = WorkingDirMock()
        self.isrunning = isrunning

    def is_running(self):
        return self.isrunning


def test_unicore_sampler():
    with pytest.raises(Exception):
        UnicoreConfig(None, 'TEST_SITE')

    with pytest.raises(Exception):
        UnicoreConfig('TEST_PRJ', '')

    with pytest.raises(Exception):
        UnicoreSampler(1, 1, UnicoreConfig('TEST_PRJ', 'TEST_SITE'))._connect_unicore()


def test_unicore_sampler_failed_auth():
    with pytest.raises(Exception):
        UnicoreSampler(1, 1, UnicoreConfig('TEST_PRJ', 'TEST_SITE'))._connect_unicore()


def test_unicore_sampler_passed_auth():
    os.environ['CLB_AUTH'] = 'test_token'
    unicore_config = UnicoreConfig('TEST_PRJ', 'TEST_SITE')

    token = UnicoreSampler(1, 1, unicore_config)._retrieve_token()
    assert token == os.environ['CLB_AUTH']
    os.environ.pop('CLB_AUTH')


def test_unicore_sampler_prepare_unicore_job():
    sampler = UnicoreSampler(1, 1, UnicoreConfig('TEST_PRJ', 'TEST_SITE'))
    job = sampler._prepare_unicore_job(Simulator())
    assert type(job) is dict
    assert all(key in job.keys() for key in ['Executable', 'Arguments', 'Project', 'Resources'])
    assert job['Executable'] == UnicoreSampler.HPC_SCRIPT
    assert len(job['Arguments']) == 4
    assert job['Project'] is not None
    assert all(key in job['Resources'].keys() for key in ['Nodes', 'NodeConstraints', 'Runtime'])


def test_unicore_sampler_stage_out_results():
    job = MockPyUnicoreJob()

    sampler = UnicoreSampler(1, 1, UnicoreConfig('TEST_PRJ', 'TEST_SITE'))
    sampler._stage_out_results(job, 'file1')


def test_unicore_sampler_stage_out_results_failed():
    job = MockPyUnicoreJob()

    sampler = UnicoreSampler(1, 1, UnicoreConfig('TEST_PRJ', 'TEST_SITE'))

    with pytest.raises(Exception):
        sampler._stage_out_results(job, 'file_that_does_not_exist')
