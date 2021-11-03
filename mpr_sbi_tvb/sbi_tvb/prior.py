from typing import Union, List
import numpy as np


class Prior:
    def __init__(self, path: str,
                 min_value: Union[int, float, List[Union[int, float]]],
                 max_value: Union[int, float, List[Union[int, float]]],
                 size: int = 1):
        self.path = path
        self.min = None
        self.max = None
        self.size = size

        if isinstance(min_value, (int, float)):
            self.min = min_value * np.ones(size)
            min_value = [min_value]
        if isinstance(max_value, (int, float)):
            self.max = max_value * np.ones(size)
            max_value = [max_value]

        if len(min_value) != len(max_value):
            raise Exception("Invalid input. Different input len.")

        if self.min is None:
            self.min = np.array(min_value)
        if self.max is None:
            self.max = np.array(max_value)
