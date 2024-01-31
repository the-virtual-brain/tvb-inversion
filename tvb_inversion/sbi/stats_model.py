from tvb_inversion.base.gen_model import StatisticalModel
from tvb_inversion.sbi.prior import PytorchPriors
from tvb.simulator.simulator import Simulator


class SBIModel(StatisticalModel):

    def __init__(
            self,
            sim: Simulator,
            params: PytorchPriors,
    ):
        super().__init__(sim, params)
