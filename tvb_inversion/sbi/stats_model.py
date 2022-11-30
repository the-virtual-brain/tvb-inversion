from tvb_inversion.base.stats_model import StatisticalModel
from tvb_inversion.sbi.prior import PytorchPrior
from tvb.simulator.simulator import Simulator


class SBIModel(StatisticalModel):

    def __init__(
            self,
            sim: Simulator,
            params: PytorchPrior,
    ):
        super().__init__(sim, params)
