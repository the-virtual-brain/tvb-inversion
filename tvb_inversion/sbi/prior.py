from typing import Union, List, Dict
import numpy as np
from torch.distributions import Distribution
from tvb_inversion.parameters import SimSeq
from tvb.simulator.simulator import Simulator

class Prior:
    def __init__(self, param: List[str], dist: Distribution):
        self.param = param
        self.dist = dist 

    def __repr__(self):
        return f'{self.param}, {self.dist}'

    def generate_sim_seq(self, sim: Simulator, num_samples: int):
        # This function supports only scalar params, for other scenarios construct the values array manually
        values = self.dist.sample((num_samples,)).numpy()
        if values.ndim == 1:
            values = values[:,np.newaxis]
        
        values = [
            [np.r_[val] for val in row]
            for row in values
        ]
        
        return SimSeq(
                template=sim,
                params=self.param,
                values=values
        )

