from tvb_inversion.base.prior import Parameters
from tvb.simulator.simulator import Simulator


class GenerativeModel:

    def __init__(
            self,
            sim: Simulator,
            params: Parameters
    ):
        self.sim = sim
        self.params = params

    def observation_model(self):
        pass
