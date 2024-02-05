from typing import List

import numpy


class Parameter:

    # TODO: To eventually inherit from HasTraits

    def __init__(self, name: str, shape: tuple = (1,),
                 value=None, min=None, max=None,
                 inds=None):  # TODO: clarify possible formats for inds
        self.name = name
        self.shape = shape
        self.value = value
        self.min = min
        self.max = max
        self.inds = inds

    def assert_shapes(self):
        if self.inds is not None:
            pass
            # TODO: find out if the indices shape corresponds to the shape
        # TODO: similarly for value, min and max, the shapes should be either the same or compatible

    def configure(self):
        self.assert_shapes()


class Parameters:

    dict = dict()

    def __init__(self, parameters: List[Parameter]):
        self.params = parameters
        self.dict = self.to_dict()

    def to_dict(self):
        d = dict()
        for p in self.params:
            d[p.name] = p
        self.dict = d
        return d

    def append_parameter(self, parameter: Parameter):
        self.params.append(parameter)
        self.dict = self.to_dict()

    def append_parameters(self, parameters: Parameters):
        for p in parameters:
            self.params.append(p)
        self.dict = self.to_dict()

    def append(self, parameter):
        if isinstance(parameter, Parameter):
            self.append_parameter(parameter)
        elif isinstance(parameter, Parameters):
            self.append_parameters(parameter)
        else:
            raise ValueError("parameter argument is neither of class Parameter nor of Parameters!")

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
