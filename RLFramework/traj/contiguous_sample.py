import numpy as np
import torch
from ..wrapper import Array
from .trajectory_element import TrajectoryElement
from .sample import Sample
from ..net import ValueNet


class ContiguousSample(Sample):
    def __init__(self, elements: list[TrajectoryElement]):
        super().__init__(elements)

    def get_GAE(self, gamma, lamda, value_net: ValueNet, max_level=None):
        states, actions, logprobs, rewards, next_states, constants = self.get_batch(gamma, level=1)

        delta = rewards.reshape(-1, 1) + constants.reshape(-1, 1) * value_net(next_states, eval=True) - value_net(states, eval=True)

        n = len(self)
        advantages = torch.zeros((n, 1)).to(value_net.device)
        running_sum = 0

        for i in range(n-1, -1, -1):
            running_sum = lamda * constants.item()[i] * running_sum + delta.item()[i]
            advantages[i] = running_sum

        return advantages


