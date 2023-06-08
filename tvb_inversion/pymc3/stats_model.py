import pymc as pm
from tvb_inversion.base.stats_model import StatisticalModel
from tvb_inversion.pymc3.prior import Pymc3Prior
from tvb.simulator.simulator import Simulator


class Pymc3Model(StatisticalModel):

    def __init__(
            self,
            sim: Simulator,
            params: Pymc3Prior,
    ):
        super().__init__(sim, params)
        self.model = params.model
