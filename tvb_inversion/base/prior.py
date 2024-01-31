from typing import List
from tvb_inversion.base.parameter import Parameter


class Prior(Parameter):

    def __init__(self, name: str, distribution,
                 shape: tuple = (1,),
                 # Perhaps to be used as initial condition, and/or storing last sample drown?:
                 value=None,
                 min=None, max=None,
                 inds=None):
        super(Prior, self).__init__(name, shape, value, min, max, inds)
        self.dist = distribution

    def __repr__(self):
        return f'{self.name}, {self.dist}'

    def sample(self, num_samples: int):
        pass

    def sample_to_numpy(self, num_samples: int):
        pass


def Priors(Parameters):

    def __init__(self, parameters: List[Prior]):
        super(Priors, self).__init__(parameters)

    def sample(self, num_samples: int):
        return [p.sample(num_samples) for p in self.params]

    def sample_to_numpy(self, num_samples: int):
        return numpy.array(self.sample(num_samples))
