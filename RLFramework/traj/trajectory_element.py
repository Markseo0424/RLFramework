import numpy as np
import torch


class TrajectoryElement(object):
    def __init__(self, state, action, logprob, reward, termination: bool, data: dict):
        if type(state) == torch.Tensor:
            state = state.detach().cpu().numpy()

        if type(action) == torch.Tensor:
            action = action.detach().cpu().numpy()

        if type(logprob) == torch.Tensor:
            logprob = logprob.detach().cpu().numpy()

        if type(reward) == torch.Tensor:
            reward = reward.detach().cpu().numpy()

        for key in data.keys():
            if type(data[key]) == torch.Tensor:
                data[key] = data[key].detach().cpu().numpy()

        self.state = state
        self.action = action
        self.logprob = logprob
        self.reward = reward
        self.termination = termination
        self.data = data

        self.next = None

    def get_estimator(self, gamma, level):
        rewards = 0
        head = self
        done = True
        for i in range(level):
            rewards += gamma ** i * head.reward

            if head.next is None or head.termination:
                done = False
                break

            head = head.next

        return rewards, head.state, gamma ** level if done else 0, head.data

    def get_generalized_estimator(self, gamma, lamda, max_level):
        rewards_list = []
        states_list = []
        constant_list = []

        data = {}

        for key in self.data.keys():
            data[key] = []

        rewards = 0
        head = self
        done = True

        for i in range(max_level):
            rewards += gamma ** i * head.reward

            if head.next is None or head.termination:
                done = False
            else:
                head = head.next

            rewards_list.append(rewards)
            states_list.append(head.state)
            constant_list.append(gamma ** (i + 1) if done else 0)

            for key in data.keys():
                data[key].append(head.data[key])

            if not done:
                break

        lambdas = lamda ** np.arange(len(rewards_list))

        gen_reward = np.dot(lambdas, np.array(rewards_list)) * (1 - lamda)
        gen_constants = lambdas * np.array(constant_list) * (1 - lamda)

        for key in data.keys():
            data[key] = np.array(data[key])

        return gen_reward, np.array(states_list), gen_constants, data
