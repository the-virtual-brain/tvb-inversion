from typing import Dict, Optional, Union, Callable

import numpy as np
import torch
import pymc3 as pm
import theano
import theano.tensor as tt

from pymc3.model import FreeRV, TransformedRV, DeterministicWrapper
from tvb.simulator.simulator import Simulator
from tvb.simulator.backend.theano import TheanoBackend
from tvb_inversion.base.base_model import StatisticalModel


class EstimatorPYMC(StatisticalModel):
    def __init__(
            self,
            sim: Simulator,
            params: Dict[str, Union[FreeRV, TransformedRV, DeterministicWrapper]],
            pymc_model: pm.Model
    ):
        super().__init__(sim, params, pymc_model)

        self.mparams = {key.split(".")[-1]: value for (key, value) in self.params.items() if "model" in key}
        self.cparams = {key.split(".")[-1]: value for (key, value) in self.params.items() if "coupling" in key}
        self.iparams = {key.split(".")[-1]: value for (key, value) in self.params.items() if "integrator" in key}

        self.dfun: Callable = self.build_dfun()

    def build_dfun(self):

        template = """
        import theano
        import theano.tensor as tt
        import numpy as np
        <%include file="theano-dfuns.py.mako"/>
        """

        dfun = TheanoBackend.build_py_func(template, content=dict(sim=self.sim), name="dfuns", print_source=False)
        return dfun

    def build_integrate(self):
        pass

    def scheme(self, x_eta, x_prev):

        dX = tt.zeros(x_prev.shape)
        cX = tt.zeros(x_prev.shape)
        parmat = self.sim.model.spatial_parameter_matrix

        dX = self.dfun(dX, x_prev, cX, parmat, **self.mparams)
        x_next = x_prev + self.sim.integrator.dt * dX + x_eta

        return x_next



