import numpy as np
import torch
from ..net import ValueNet
from ..wrapper import Array
from .trajectory_element import TrajectoryElement


class Sample(object):
    def __init__(self, elements: list[TrajectoryElement]):
        self.__elements = elements
        self.__batches = None
        self.__level = None

    def __len__(self):
        return len(self.__elements)

    def get_batch(self, gamma, level=1):
        if self.__level == level and self.__batches is not None:
            return self.__batches

        states = []
        actions = []
        logprobs = []
        rewards = []
        next_states = []
        constants = []
        data = {}
        next_data = {}

        first_element = self.__elements[0]
        for key in first_element.data.keys():
            data[key] = []
            next_data[key] = []

        for element in self.__elements:
            states.append(element.state)
            actions.append(element.action)
            logprobs.append(element.logprob)

            reward, next_state, constant, next_data_ = element.get_estimator(gamma=gamma, level=level)

            rewards.append(reward)
            next_states.append(next_state)
            constants.append(constant)

            for key in data.keys():
                data[key].append(element.data[key])
                next_data[key].append(next_data_)

        for key in data.keys():
            data[key] = np.array(data[key])
            next_data[key] = np.array(next_data[key])

        self.__batches = (Array(states, data=data), Array(actions), Array(logprobs),
                          Array(rewards), Array(next_states, data=next_data), Array(constants))

        return self.__batches

    def get_GAE(self, gamma, lamda, value_net: ValueNet, max_level=10):
        states, actions, logprobs, rewards, next_states, constants = self.get_batch(gamma, level=1)

        n = len(self)
        advantages = torch.zeros((n, 1)).to(value_net.device)

        for i, element in enumerate(self.__elements):
            rewards, next_states, constants, data = element.get_generalized_estimator(gamma, lamda, max_level=max_level)
            advantage = Array(rewards.reshape(-1,1)) + torch.sum(
                            (value_net(Array(next_states, data=data)) * Array(constants.reshape(-1, 1))).item())
            advantages[i] = advantage.item()

        advantages = advantages - value_net(states)

        return advantages
