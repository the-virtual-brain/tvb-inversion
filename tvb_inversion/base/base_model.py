from typing import Dict, Union, Optional

import numpy as np
import torch
import pymc3 as pm
import theano
import theano.tensor as tt
from pymc3.model import FreeRV, TransformedRV, DeterministicWrapper
from tvb.simulator.simulator import Simulator


class StatisticalModel:
    def __init__(
            self,
            sim: Simulator,
            params: Dict[str, Union[FreeRV, TransformedRV, DeterministicWrapper, np.ndarray, torch.Tensor, torch.distributions.Distribution]],
            pymc_model: Optional[pm.Model] = None
    ):
        self.sim = sim
        self.params = params
        if pymc_model is not None:
            self.pymc_model = pymc_model

    def observation_model(self):
        pass
