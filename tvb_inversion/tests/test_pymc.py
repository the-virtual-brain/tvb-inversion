import unittest
from typing import Callable

from tvb_inversion.pymc.examples import (
    create_2node_simulator,
    default_model_builders,
    uninformative_model_builders,
    custom_model_builders
)


class BaseTestPymc3Inference(unittest.TestCase):
    simulation_length = 10
    sample_kwargs = dict(draws=25, tune=25, cores=1)

    def run_test(self, model_builder: Callable):
        sim = create_2node_simulator(simulation_length=self.simulation_length)
        (t, X), = sim.run()
        # self.assertRaises(ValueError, self.model_builder, sim=sim, observation=X, **self.sample_kwargs)
        _ = model_builder(sim=sim, observation=X, **self.sample_kwargs)


class TestDefaultModelBuilders(BaseTestPymc3Inference):

    def test(self):
        self.run_test(default_model_builders)


class TestUninformativeModelBuilders(BaseTestPymc3Inference):

    def test(self):
        self.run_test(uninformative_model_builders)


class TestCustomModelBuilders(BaseTestPymc3Inference):

    def test(self):
        self.run_test(custom_model_builders)


