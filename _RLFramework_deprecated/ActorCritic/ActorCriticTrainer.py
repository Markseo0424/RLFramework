import abc
import torch

from _RLFramework_deprecated.RLTrainer import RLTrainer
from _RLFramework_deprecated.Network import Network
from _RLFramework_deprecated.Environment import Environment
from _RLFramework_deprecated.Agent import Agent


class ActorCriticTrainer(RLTrainer):
    def __init__(self, policy_net: Network, value_net: Network, environment: Environment, agent: Agent,
                 gamma=1, epsilon=1e-5, grad_clip=None):
        """
        :param gamma: Discount factor of Reward
        :param epsilon: Small value that is added for log calculation.
        :param grad_clip: Gradient Clipping norm magnitude.
        """
        super().__init__(environment=environment, agent=agent)

        self.policy_net = policy_net
        self.value_net = value_net
        self.gamma = gamma
        self.eps = epsilon
        self.grad_clip = grad_clip

    def train(self, state, action, reward, next_state):
        """
        :param state: Current state of an environment.
        :param action: Current action of an agent.
        :param reward: Current reward of current state-action set.
        :param next_state: Next state of an environment.
        :return A tuple of (policy loss, value loss) of the train step.
        Train by Actor-Critic method.
        Expect unbatched input.
        """
        if next_state is None:
            pred_next_value = torch.tensor([0])
        else:
            pred_next_value = self.value_net.predict(next_state)

        target_value = reward + self.gamma * pred_next_value
        target_value = target_value.cpu()

        policy_loss_function = lambda p, t: -torch.log(p[action] + self.eps) * t
        value_loss_function = lambda p, t: torch.pow(torch.norm(p - t), 2) / 2

        policy_loss = self.policy_net.train_batch(state, target_value, loss_function=policy_loss_function, clip=self.grad_clip)
        value_loss = self.value_net.train_batch(state, target_value, loss_function=value_loss_function, clip=self.grad_clip)

        return policy_loss, value_loss

    def check_train(self):
        return True

    @abc.abstractmethod
    def check_reset(self):
        """
        :return: Bool value for whether to reset an Environment.
        """
        pass

    @abc.abstractmethod
    def reset_params(self):
        pass
