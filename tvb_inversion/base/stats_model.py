import numpy as np
from tvb_inversion.base.prior import Prior
from tvb_inversion.base.parameters import SimSeq
from tvb.simulator.simulator import Simulator


class StatisticalModel:

    def __init__(
            self,
            sim: Simulator,
            params: Prior
    ):
        self.sim = sim
        self.params = params

    def observation_model(self):
        pass

    def generate_sim_seq(self, num_samples: int):
        # This function supports only scalar params, for other scenarios construct the values array manually
        values = self.params.sample_to_numpy(num_samples)
        if values.ndim == 1:
            values = values[:, np.newaxis]

        values = [
            [np.r_[val] for val in row]
            for row in values
        ]

        return SimSeq(
            template=self.sim,
            params=self.params.names,
            values=values
        )
