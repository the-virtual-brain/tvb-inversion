from typing import Dict, Optional, Union, Callable, List

import numpy as np
import torch
import pymc3 as pm
import theano
import theano.tensor as tt

from tvb_inversion.base.inference import Estimator
from tvb_inversion.pymc3.stats_model import Pymc3Model
from tvb_inversion.base.metrics import Metric
from tvb.simulator.backend.theano import TheanoBackend


class EstimatorPYMC(Estimator):

    def __init__(
            self,
            stats_model: Pymc3Model,
            metrics: Optional[List[Metric]] = None
    ):

        super().__init__(stats_model, metrics)

        self.mparams = {n.split(".")[-1]: d for (n, d) in zip(self.prior.names, self.prior.dist) if "model" in n}
        self.cparams = {n.split(".")[-1]: d for (n, d) in zip(self.prior.names, self.prior.dist) if "coupling" in n}
        self.iparams = {n.split(".")[-1]: d for (n, d) in zip(self.prior.names, self.prior.dist) if "integrator" in n}

        self.dfun: Callable = self.build_dfun()
        self.cfun: Callable = self.build_cfun()

    @property
    def model(self):
        return self.stats_model.model

    def build_dfun(self):

        template = """
        import theano
        import theano.tensor as tt
        import numpy as np
        <%include file="theano-dfuns.py.mako"/>
        """

        dfun = TheanoBackend.build_py_func(template, content=dict(sim=self.sim), name="dfun", print_source=False)
        return dfun

    def build_cfun(self):

        template = """
        import theano
        import theano.tensor as tt
        import numpy as np
        <%include file="theano-coupling.py.mako"/>
        """

        cfun = TheanoBackend.build_py_func(template, content=dict(sim=self.sim), name="coupling", print_source=False)
        return cfun

    def build_integrator(self):
        pass

    def scheme(self, x_eta, x_prev):

        cX = tt.zeros(x_prev.shape)
        cX = self.cfun(cX, self.sim.connectivity.weights, x_prev, self.sim.connectivity.delay_indices, **self.cparams)

        dX = tt.zeros(x_prev.shape)
        parmat = self.sim.model.spatial_parameter_matrix
        dX = self.dfun(dX, x_prev, cX, parmat, **self.mparams)

        x_next = x_prev + self.sim.integrator.dt * dX + x_eta

        return x_next



