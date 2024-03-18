import torch
import numpy as np
from .space import Space


class Discrete(Space):
    def __init__(self, n: int, start: int = 0):
        super().__init__()
        self.n = n
        self.shape = (n,)
        self.start = start

    def sample(self, a=1):
        if a > 1:
            return np.random.randint(self.n, size=a) + self.start
        else:
            return np.random.randint(self.n) + self.start

    def greedy(self, x):
        assert x.shape[-1] == self.n, f"shape mismatch! input must be length of {self.n}"

        if len(x.shape) == 1:
            unbatched = True
            x = x.reshape(1, self.n)
        else:
            unbatched = False

        if type(x) == np.ndarray:
            res = np.argmax(x, axis=1) + self.start
        elif type(x) == torch.Tensor:
            res = torch.argmax(x, dim=1) + self.start
        else:
            assert False, "input must be numpy.ndarray or torch.Tensor."

        if unbatched:
            return res[0]
        else:
            return res
