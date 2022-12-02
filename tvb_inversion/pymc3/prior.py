from typing import List, Union
from pymc3.model import FreeRV, TransformedRV, DeterministicWrapper
from tvb_inversion.base.prior import Prior


class Pymc3Prior(Prior):

    def __init__(self, names: List[str], dist: List[Union[FreeRV, TransformedRV, DeterministicWrapper]]):
        super().__init__(names, dist)

    def sample(self, num_samples: int):
        return [d.distribution.random(size=(num_samples, )) for d in self.dist]

    def sample_to_numpy(self, num_samples: int):
        self.sample(num_samples)

    def to_dict(self):
        return {n: d for (n, d) in zip(self.names, self.dist)}