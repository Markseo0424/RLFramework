import abc
import copy
import numpy as np
import torch
import torch.nn as nn

from RLFramework.RLTrainer import RLTrainer
from RLFramework.DQN.ReplayBuffer import ReplayBuffer
from RLFramework.Network import Network
from RLFramework.Environment import Environment
from RLFramework.Agent import Agent


class DQNTrainer(RLTrainer):
    def __init__(self, q_net: Network, environment: Environment, agent: Agent, alpha=1, gamma=1, grad_clip=1,
                 loss_function=nn.MSELoss(), use_replay_buffer=True, use_target_q=True,
                 start_train=5000, train_freq=4, target_update_freq=1000, batch_size=64,
                 buffer_len=1000000, slot_weights: dict = None):
        """
        :param alpha: Reward rate when calculating target Q value.
        :param gamma: Discount factor of Reward
        :param grad_clip: Gradient Clipping norm magnitude.
        :param start_train: Step to start train.
        :param batch_size: Batch size of replay buffer.
        :param buffer_len: Max length of each replay buffer slots.
        :param slot_weights: Definition of replay buffer slots and their weights.
        """
        super().__init__(environment=environment, agent=agent)

        assert target_update_freq % train_freq == 0, "target_update_freq must be multiple of train_freq."

        self.q_net = q_net

        if use_target_q:
            self.target_q_net = copy.deepcopy(self.q_net)
        else:
            self.target_q_net = self.q_net

        self.alpha = alpha
        self.gamma = gamma
        self.grad_clip = grad_clip
        self.loss_function = loss_function

        self.start_train = start_train
        self.train_freq = train_freq
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size

        self.use_replay_buffer = use_replay_buffer
        self.replay_buffer = ReplayBuffer(buffer_len=buffer_len, slot_weights=slot_weights)

    def memory(self):
        """
        Saves data of (state, action, reward, next state) to the replay buffer.
        Can be overridden when need to memorize other values.
        """
        if self.use_replay_buffer:
            if self.environment.timestep >= 1:
                state, action, reward, next_state = self.memory_state[-2], self.memory_action[-2], self.memory_reward[
                    -1], self.memory_state[-1]
                self.replay_buffer.append(state, action, reward, next_state,
                                          slot=self.choose_slot(state, action, reward, next_state))

    def choose_slot(self, state, action, reward, next_state):
        """
        :param state: Current state of environment.
        :param action: Current action of agent.
        :param reward: Reward of Current state-action set.
        :param next_state: Next state of environment.
        :return: Slot name where this data would be inserted.
        Check state, action and reward, and returns replay buffer slot where the data should be inserted.
        """
        return "default"

    def train(self, state, action, reward, next_state):
        """
        :param state: Current state of an environment.
        :param action: Current action of an agent.
        :param reward: Current reward of current state-action set.
        :param next_state: Next state of an environment.
        :return Loss of the train step.
        Train by DQN method.
        Expect unbatched input.
        """
        if not self.use_replay_buffer:
            current_Q = self.q_net.predict(state)

            if next_state is None:
                next_max_Q = 0
            else:
                next_max_Q = torch.max(self.target_q_net.predict(next_state))

            target_Q = current_Q
            target_Q[action] += self.alpha * reward + self.gamma * (next_max_Q - current_Q[action])

            loss = self.q_net.train_batch(state, target_Q.cpu().detach().numpy(), self.loss_function, self.grad_clip)

        else:
            batches = self.replay_buffer.sample(self.batch_size)

            x = []
            y = []

            for _state, _action, _reward, _next_state in batches:
                current_Q = self.q_net.predict(_state)

                if _next_state is None:
                    next_max_Q = 0
                else:
                    next_max_Q = torch.max(self.target_q_net.predict(_next_state))

                target_Q = current_Q
                target_Q[_action] += self.alpha * (_reward + self.gamma * next_max_Q - current_Q[_action])

                x.append(_state)
                y.append(target_Q.cpu().detach().numpy())

            loss = self.q_net.train_batch(np.stack(x, axis=0), np.stack(y, axis=0), self.loss_function, self.grad_clip)

            if self.timestep % self.target_update_freq == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())

        return loss

    def check_train(self):
        """
        :return: Bool value for whether to train.
        """
        if not self.use_replay_buffer:
            return True

        return self.timestep >= self.start_train and self.timestep % self.train_freq == 0

    @abc.abstractmethod
    def check_reset(self):
        """
        :return: Bool value for whether to reset an Environment.
        """
        pass

    @abc.abstractmethod
    def reset_params(self):
        pass
