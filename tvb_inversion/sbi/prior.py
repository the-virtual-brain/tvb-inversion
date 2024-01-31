from typing import List
from torch.distributions import Distribution
from tvb_inversion.base.prior import Prior


class PytorchPrior(Prior):

    def __init__(self,
                 name: str,
                 distribution: Distribution,
                 shape: tuple = (1,),
                 # Perhaps to be used as initial condition, and/or storing last sample drown?:
                 value=None,
                 min=None, max=None,
                 inds=NoneDistribution):
        super(PytorchPrior, self).__init__(name, shape, distribution, value, min, max, inds)

    def sample(self, num_samples: int):
        return self.dist.sample((num_samples,))

    def sample_to_numpy(self, num_samples: int):
        return self.sample(num_samples).numpy()


class PytorchPriors(Priors):

    def __init__(self, parameters: List[PytorchPrior]):
        super(PytorchPriors, self).__init__(parameters)
