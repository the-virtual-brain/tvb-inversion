import numpy as np
from tvb_inversion.base.prior import Priors
from gen_model import GenerativeModel
from tvb_inversion.base.sim_seq import SimSeq
from tvb.simulator.simulator import Simulator


class StatisticalModel(GenerativeModel):

    def __init__(
            self,
            sim: Simulator,
            params: Priors
    ):
        super(StatisticalModel, self).__init__(sim, params)

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
