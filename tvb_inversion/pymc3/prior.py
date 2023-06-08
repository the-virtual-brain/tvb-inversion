from typing import List, Union
from pymc import Model
# from pymc.model import FreeRV, TransformedRV, DeterministicWrapper
from tvb_inversion.base.prior import Prior


class Pymc3Prior(Prior):

    model: Model

    def __init__(self, model: Model, names: List[str],
                 dist: List):
                 # dist: List[Union[FreeRV, TransformedRV, DeterministicWrapper]]):
        self.model = model
        super().__init__(names, dist)
        self.dict = self.to_dict()

    def sample(self, num_samples: int):
        return [d.distribution.random(size=(num_samples, )) for d in self.dist]

    def sample_to_numpy(self, num_samples: int):
        self.sample(num_samples)

    def append(self, names, dist):
        self.names.extend(list(names))
        self.dist.extend(list(dist))
        self.dict = self.to_dict()
        return self

    def to_dict(self):
        return dict(zip(self.names, self.dist))

    def get_params_from_path(self, param_type):
        return {pname.split(".")[-1]: pval for pname, pval in self.dict.items() if param_type in pname}

    def get_model_params(self):
        return self.get_params_from_path("model")

    def get_coupling_params(self):
        return self.get_params_from_path("coupling")

    def get_integrator_params(self):
        return self.get_params_from_path("integrator")

    def get_monitor_params(self, id=0):
        return self.get_params_from_path("monitors[%d]" % id)

    def get_observation_model_params(self):
        return self.get_params_from_path("observation.")
