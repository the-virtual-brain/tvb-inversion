from typing import List, Union
from pymc import Model
# from pymc.model import FreeRV, TransformedRV, DeterministicWrapper
from tvb_inversion.base.prior import Prior, Priors


class Pymc3Prior(Prior):

    model: Model

    def __init__(self, model: Model,
                 name: str,
                 distribution,  # Union[FreeRV, TransformedRV, DeterministicWrapper]]
                 shape: tuple = (1,),
                 # Perhaps to be used as initial condition, and/or storing last sample drown?:
                 value=None,
                 min=None, max=None,
                 inds=None):
        self.model = model
        super(Pymc3Prior, self).__init__(name, shape, distribution, value, min, max, inds)

    def sample(self, num_samples: int):
        return self.dist.distribution.random(size=(num_samples, ))

    def sample_to_numpy(self, num_samples: int):
        return self.sample(num_samples)


class Pymc3Priors(Priors):

    model: Model

    def __init__(self, parameters: List[Pymc3Prior]):
        super(Pymc3Priors, self).__init__(parameters)
        self.model = self.parameters[0].model
