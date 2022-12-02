from copy import deepcopy
from dataclasses import dataclass
from typing import List, Any, Optional
from tvb.simulator.simulator import Simulator


class ParamGetter:
    pass


@dataclass
class SimSeq:
    "A sequence of simulator configurations."

    template: Simulator
    params: List[str]
    values: List[List[Any]]
    getters: Optional[List[Optional[ParamGetter]]] = None  # is the first Optional needed?
    # TODO consider transpose, so a names can have a remote data source
    # to load when constructing the sequence

    def __iter__(self):
        self.pos = 0
        return self

    def __post_init__(self):
        self.template.configure()  # deepcopy doesn't work on un-configured simulator o_O
        if self.getters is None:
            self.getters = [None]*len(self.params)
        else:
            assert len(self.getters) == len(self.params)

    def __next__(self):
        if self.pos >= len(self.values):
            raise StopIteration
        obj = deepcopy(self.template)
        updates = zip(self.params, self.getters, self.values[self.pos])
        for key, getter, val in updates:
            if getter is not None:
                val = getter(val)
            exec(f'obj.{key} = val',
                 {'obj': obj, 'val': val})
        self.pos += 1
        return obj