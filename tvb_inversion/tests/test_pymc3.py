import unittest
from typing import Callable

from tvb_inversion.pymc3.examples import (
    create_2node_simulator,
    default_model_builders,
    uninformative_model_builders,
    custom_model_builders
)


class BaseTestPymc3Inference(unittest.TestCase):
    simulation_length = 10
    sample_kwargs = dict(draws=25, tune=25, cores=1)
    model_builder: Callable = None

    def run_test(self):
        sim = create_2node_simulator(simulation_length=self.simulation_length)
        (t, X), = sim.run()
        _ = self.model_builder(sim=sim, observation=X, **self.sample_kwargs)


class TestDefaultModelBuilders(BaseTestPymc3Inference):
    model_builder = default_model_builders

    def test(self):
        self.run_test()


class TestUninformativeModelBuilders(BaseTestPymc3Inference):
    model_builder = uninformative_model_builders

    def test(self):
        self.run_test()


class TestCustomModelBuilders(BaseTestPymc3Inference):
    model_builder = custom_model_builders

    def test(self):
        self.run_test()


