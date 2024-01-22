import abc
import copy

import numpy as np
import torch
import torch.nn as nn
from RLFramework.RLTrainer import RLTrainer
from RLFramework.DQN.ReplayBuffer import ReplayBuffer
from RLFramework.Network import Network


class DQNTrainer(RLTrainer):
    def __init__(self, network: Network, *args, alpha=1, gamma=1, use_replay_buffer=True, use_target_q=True,
                 start_train=5000, train_freq=4, target_update_freq=1000, batch_size=64,
                 buffer_len=1000000, slot_weights: dict = None, **kwargs):
        super().__init__(*args, **kwargs)

        assert target_update_freq % train_freq == 0, "target_update_freq must be multiple of train_freq."

        self.qnet = network

        if use_target_q:
            self.tqnet = copy.deepcopy(self.qnet)
        else:
            self.tqnet = self.qnet

        self.alpha = alpha
        self.gamma = gamma
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
        Function that progress training based on DQN method.
        """
        if not self.use_replay_buffer:
            current_Q = self.qnet.predict(state)
            next_max_Q = torch.max(self.tqnet.predict(next_state))

            target_Q = current_Q
            target_Q[action] += self.alpha * reward + self.gamma * (next_max_Q - current_Q[action])

            loss = self.qnet.train_batch(state, target_Q, nn.MSELoss(), 1)

        else:
            batches = self.replay_buffer.sample(self.batch_size)

            x = []
            y = []

            for _state, _action, _reward, _next_state in batches:
                current_Q = self.qnet.predict(_state)
                next_Q = self.tqnet.predict(_next_state)

                if next_Q is None:
                    next_max_Q = 0
                else:
                    next_max_Q = torch.max(next_Q)

                target_Q = current_Q
                target_Q[0, _action] += self.alpha * (_reward + self.gamma * next_max_Q - current_Q[0, _action])

                x.append(_state)
                y.append(target_Q.cpu().detach().numpy())

            loss = self.qnet.train_batch(np.concatenate(x, axis=0), np.concatenate(y, axis=0), nn.MSELoss(), 1)

            if self.timestep % self.target_update_freq == 0:
                self.tqnet.load_state_dict(self.qnet.state_dict())

        return loss.item()

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
