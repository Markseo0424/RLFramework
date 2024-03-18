import numpy as np


class Space(object):
    def __int__(self):
        super().__init__()
        self.shape = None

    def sample(self, a=1):
        raise NotImplementedError

    def greedy(self, x):
        raise NotImplementedError
