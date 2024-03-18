import numpy as np
from .space import Space


class Continuous(Space):
    def __init__(self, upper: np.ndarray, lower: np.ndarray):
        assert upper.shape == lower.shape, "upper and lower should have same shape."
        super().__init__()

        self.shape = upper.shape
        self.upper = upper
        self.lower = lower

    def sample(self, a=1):
        if a > 1:
            return np.random.sample((a, *self.shape)) * (self.upper - self.lower).reshape(1, -1) + self.lower.reshape(1, -1)
        else:
            return np.random.sample(self.shape) * (self.upper - self.lower) + self.lower

    def greedy(self, x):
        return x
