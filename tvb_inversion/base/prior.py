from typing import List


class Prior:
    def __init__(self, names: List[str], dist):
        self.names = names
        self.dist = dist

    def __repr__(self):
        return f'{self.names}, {self.dist}'

    def sample(self, num_samples: int):
        pass

    def sample_to_numpy(self, num_samples: int):
        pass
