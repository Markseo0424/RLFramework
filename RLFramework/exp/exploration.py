import numpy as np
from ..space import Space


class Exploration(object):
    def __init__(self, action_space: Space = None):
        super().__init__()

        if action_space is not None:
            self.init_space(action_space)
        else:
            self.action_space = None
            self.policy_shape = None

    def init_space(self, action_space: Space):
        self.action_space = action_space
        self.policy_shape = self.get_policy_shape(self.action_space)

    def get_policy_shape(self, action_space: Space):
        return action_space.shape

    def explore(self, x, numpy=False):
        return self.action_space.greedy(x)

    def greedy(self, x, numpy=False):
        return self.action_space.greedy(x)

    def get_logprob(self, x, action, numpy=False):
        return 0, 0

    def decay(self):
        pass

    def reset(self):
        pass
