import pymc as pm
from tvb_inversion.base.stats_model import StatisticalModel
from tvb_inversion.pymc.prior import PymcPrior
from tvb.simulator.simulator import Simulator


class PymcModel(StatisticalModel):

    def __init__(
            self,
            sim: Simulator,
            params: PymcPrior,
    ):
        super().__init__(sim, params)
        self.model = params.model
