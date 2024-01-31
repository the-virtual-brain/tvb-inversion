import pymc as pm
from tvb_inversion.base.gen_model import StatisticalModel
from tvb_inversion.pymc3.prior import Pymc3Priors
from tvb.simulator.simulator import Simulator


class Pymc3Model(StatisticalModel):

    def __init__(
            self,
            sim: Simulator,
            params: Pymc3Priors,
    ):
        super(Pymc3Model, self).__init__(sim, params)
        self.model = params.model
