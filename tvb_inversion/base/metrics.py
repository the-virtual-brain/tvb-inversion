import numpy as np


class Metric:
    "A summary statistic for a simulation."
    def __call__(self, t, y) -> np.ndarray: # what about multi metric returning dict of statistics? Also, chaining?
        pass


class NodeVariability(Metric):
    "A simplistic simulation statistic."
    def __call__(self, t, y):
        return np.std(y[t > (t[-1] / 2), 0, :, 0], axis=0)