from typing import List
from torch.distributions import Distribution
from tvb_inversion.base.prior import Prior


class PytorchPrior(Prior):

    def __init__(self, names: List[str], dist: Distribution):
        super().__init__(names, dist)

    def sample(self, num_samples: int):
        return self.dist.sample((num_samples,))

    def sample_to_numpy(self, num_samples: int):
        return self.sample(num_samples).numpy()
